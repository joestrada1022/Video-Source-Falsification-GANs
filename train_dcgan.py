import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import TensorBoard

from pathlib import Path
import argparse, os

from utils.datagenGAN import DataSetGeneratorGAN
from utils.datagenGAN import DataGeneratorGAN
from utils.callbacks import GANMonitor, ModelSaveCallback
from models import Generator, Discriminator, DCGAN

parser = argparse.ArgumentParser(description="Train a WGAN model")
parser.add_argument("--data_path", type=str, required=True, help="Path to the data folder")
parser.add_argument("--classifier_path", type=str, required=True, help="Path to the external classifier")
parser.add_argument("--use_cpu", type=bool, default=False, help="Use CPU for training")
parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder")

if __name__ == "__main__":

    EPOCHS = 30
    BATCH_SIZE = 12

    # parse arguments
    args = parser.parse_args()
    dataset_path = args.data_path
    image_path = args.output_path + "/images"
    model_path = args.output_path + "/models"
    tensor_board_path = args.output_path + "/logs"
    classifier_path = args.classifier_path
    use_cpu = args.use_cpu

    # set CPU
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # create dataset
    dataset_maker = DataSetGeneratorGAN(dataset_path)

    train_set = dataset_maker.create_dataset()
    print(f"Train dataset contains {len(train_set)} samples")

    num_classes = len(dataset_maker.get_class_names())

    shape = (1080 // 3, 1920 // 3, 3)

    # define models
    gen = Generator(shape, num_classes)
    gen.create_model()

    disc = Discriminator(shape)
    disc.create_model()

    # load pre-trained classifier
    print('Classifier Exists: ', os.path.exists(classifier_path))
    classifier = keras.models.load_model(classifier_path)

    # create callbacks
    image_callback = GANMonitor(data_path=dataset_path, save_path=image_path)
    model_callback = ModelSaveCallback(gen.model, disc.model, model_path)
    tensorboard_callback = TensorBoard(log_dir=tensor_board_path)

    # create optimizers
    generator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    # compile and train
    dcgan = DCGAN(discriminator=disc.model, generator=gen.model, classifier=classifier, input_shape=shape, num_classes=num_classes, embedding_dim=50)

    dcgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
    )

    dcgan.fit(
        DataGeneratorGAN(train_set, num_classes, BATCH_SIZE),
        epochs=EPOCHS,
        initial_epoch=0,
        callbacks=[model_callback, tensorboard_callback, image_callback],
    )

    gen.model.save(model_path + "/final_gen.keras")
    disc.model.save(model_path + "/final_disc.keras")

    print("Training complete")

