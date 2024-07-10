import tensorflow as tf
import keras
from keras import optimizers # type: ignore
from keras.callbacks import TensorBoard # type: ignore

import argparse
import os

from generator import Generator
from discriminator import Discriminator
from datagenGAN import DataSetGeneratorGAN
from datagenGAN import DataGeneratorGAN
from wgan import WGAN
from callbacks import GANMonitor, ModelSaveCallback

parser = argparse.ArgumentParser(description="Train a WGAN model")
parser.add_argument("--data_path", type=str, required=True, help="Path to the data folder")
parser.add_argument("--image_path", type=str, required=True, help="Path to the image callback output folder")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model output folder")
parser.add_argument("--tensorboard_path", type=str, required=True, help="Path to the tensorboard output")
parser.add_argument("--use_cpu", type=bool, default=False, help="Use CPU for training")

if __name__ == "__main__":

    EPOCHS = 30
    BATCH_SIZE = 12

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
    gen.print_model_summary()

    disc = Discriminator(shape)
    disc.create_model()
    disc.print_model_summary()

    exit()

    # create callbacks
    image_callback = GANMonitor(data_path=dataset_path, save_path=image_path)
    model_callback = ModelSaveCallback(gen.model, disc.model, model_path)
    tensorboard_callback = TensorBoard(log_dir=tensor_board_path)

    # create optimizers
    generator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    # compile and train
    wgangp = WGAN(discriminator=disc.model, generator=gen.model, input_shape=shape)

    wgangp.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
    )

    wgangp.fit(
        DataGeneratorGAN(train_set, num_classes, BATCH_SIZE),
        epochs=EPOCHS,
        initial_epoch=0,
        callbacks=[model_callback, tensorboard_callback, image_callback],
    )

    gen.model.save(model_path + "/final_gen.keras")
    disc.model.save(model_path + "/final_disc.keras")

    print("Training complete")
