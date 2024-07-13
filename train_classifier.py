import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import TensorBoard

from pathlib import Path
import argparse, os

from scd_videos.models import MobileNet
from utils.datagenGAN import DataSetGeneratorGAN
from utils.datagenGAN import DataGeneratorGAN
from utils.callbacks import ModelSaveCallback

parser = argparse.ArgumentParser(description="Train a Classifier model")
parser.add_argument(
    "--data_path", type=str, required=True, help="Path to the data folder"
)
parser.add_argument(
    "--model_path", type=str, required=True, help="Path to the model output folder"
)
parser.add_argument(
    "--tensorboard_path", type=str, required=True, help="Path to the tensorboard output"
)
parser.add_argument("--use_cpu", type=bool, default=False, help="Use CPU for training")

if __name__ == "__main__":

    EPOCHS = 20
    BATCH_SIZE = 32

    # parse arguments
    args = parser.parse_args()
    dataset_path = args.data_path
    model_path = args.model_path
    tensor_board_path = args.tensorboard_path
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
    val_set = dataset_maker.create_dataset('Validation')
    print(f"Validation dataset contains {len(val_set)} samples")
    test_set = dataset_maker.create_dataset('Testing')
    print(f"Test dataset contains {len(test_set)} samples")

    num_classes = len(dataset_maker.get_class_names())

    shape = (1080 // 3, 1920 // 3, 3)

    # load the external classifier
    net = MobileNet(
        global_results_dir=Path("."), model_name="MobileNet", const_type=None, lr=0.0001
    )
    classifier = net.create_model(num_classes=num_classes, height=shape[0], width=shape[1], use_pretrained=True)

    classifier_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    classifier.compile(optimizer=classifier_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # create callbacks
    model_callback = ModelSaveCallback(generator=None, discriminator=classifier, save_path=model_path)
    tensorboard_callback = TensorBoard(log_dir=tensor_board_path)

    classifier.fit(DataGeneratorGAN(train_set, num_classes, BATCH_SIZE), validation_data=DataGeneratorGAN(val_set, num_classes, BATCH_SIZE), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[model_callback, tensorboard_callback])

    classifier.save(model_path + '/final_class.keras')

    print("Training completed")

    # evaluate the model
    print("Evaluating the model")
    classifier.evaluate(DataGeneratorGAN(test_set, num_classes, BATCH_SIZE))
