import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import TensorBoard

from pathlib import Path
import argparse, os

from models.cgan import CGAN
from utils.datagenGAN import DataSetGeneratorGAN
from utils.datagenGAN import DataGeneratorGAN
from utils.callbacks import GANMonitor, ModelSaveCallback
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scd_videos.models import MobileNet

parser = argparse.ArgumentParser(description="Train a WGAN model")
parser.add_argument(
    "--data_path", type=str, required=True, help="Path to the data folder"
)
parser.add_argument(
    "--image_path",
    type=str,
    required=True,
    help="Path to the image callback output folder",
)
parser.add_argument(
    "--model_path", type=str, required=True, help="Path to the model output folder"
)
parser.add_argument(
    "--tensorboard_path", type=str, required=True, help="Path to the tensorboard output"
)
parser.add_argument("--use_cpu", type=bool, default=False, help="Use CPU for training")

if __name__ == "__main__":

    EPOCHS = 30
    BATCH_SIZE = 24

    # parse arguments
    args = parser.parse_args()
    dataset_path = args.data_path
    model_path = args.model_path
    tensor_board_path = args.tensorboard_path
    image_path = args.image_path
    use_cpu = args.use_cpu

    # set CPU
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # create dataset
    dataset_maker = DataSetGeneratorGAN(dataset_path)

    train_set = dataset_maker.create_dataset()
    print(f"Train dataset contains {len(train_set)} samples")

    num_classes = len(dataset_maker.get_class_names())

    shape = (1080 // 3, 1920 // 3, 3)

# load the generator model
original_generator = keras.models.load_model('generated/models/wcgan-gp7/generator_epoch_23.keras')

# load the external classifier
net = MobileNet(
    global_results_dir=Path("."), model_name="MobileNet", const_type=None, lr=0.1
)
classifier = net.create_model(num_classes=num_classes, height=shape[0], width=shape[1], use_pretrained=True)

cgan = CGAN(generator=original_generator, classifier=classifier, input_shape=shape, num_classes=num_classes)

generator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
classifier_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

cgan.compile(g_optimizer=generator_optimizer, c_optimizer=classifier_optimizer)

# create callbacks
image_callback = GANMonitor(data_path=dataset_path, save_path=image_path)
model_callback = ModelSaveCallback(generator=cgan.get_generator(), discriminator=cgan.get_classifier(), save_path=model_path)
tensorboard_callback = TensorBoard(log_dir=tensor_board_path)

cgan.fit(DataGeneratorGAN(train_set, num_classes, BATCH_SIZE), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[image_callback, model_callback, tensorboard_callback])

cgan.get_generator().save(model_path + '/final_gen.keras')
cgan.get_classifier().save(model_path + '/final_class.keras')

