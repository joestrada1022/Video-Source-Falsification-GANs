import tensorflow as tf
import numpy as np
from tensorflow.keras.constraints import Constraint # type: ignore
from tensorflow.keras import optimizers # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore
from tensorflow.keras.models import load_model # type: ignore

import argparse
import os

from generator import Generator
from discriminator import Discriminator
from datagenGAN import DataSetGeneratorGAN
from datagenGAN import DataGeneratorGAN
from wgan import WGAN, GANMonitor, ModelSaveCallback
import keras

parser = argparse.ArgumentParser(description="Train a WGAN model")
parser.add_argument("--data_path", type=str, required=True, help="Path to the data folder")
parser.add_argument("--image_path", type=str, required=True, help="Path to the image callback output folder")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model output folder")
parser.add_argument("--tensorboard_path", type=str, required=True, help="Path to the tensorboard output")
parser.add_argument("--use_gpu", type=bool, default=False, help="Use GPU for training")

if __name__ == "__main__":

    EPOCHS = 4
    BATCH_SIZE = 32

    # parse arguments
    args = parser.parse_args()
    dataset_path = args.data_path
    model_path = args.model_path
    tensor_board_path = args.tensorboard_path
    image_path = args.image_path
    use_gpu = args.use_gpu

    # set GPU
    if use_gpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # create dataset
    dataset_maker = DataSetGeneratorGAN(dataset_path)

    train = dataset_maker.create_dataset()
    print(f"Train dataset contains {len(train)} samples")

    num_classes = len(dataset_maker.get_class_names())

    shape = (1080 // 4, 1920 // 4, 3)

    # define models
    gen = Generator(shape, num_classes)
    gen.create_model()
    gen.print_model_summary()

    disc = Discriminator(shape)
    disc.create_model()
    disc.print_model_summary()

    # load pre-trained classifier
    path = 'scd-videos/saved_model'
    print('File Exists: ', os.path.exists(path))
    classifier = keras.layers.TFSMLayer(path, call_endpoint='serving_default')

    # create callbacks
    image_callback = GANMonitor(save_path=image_path)
    model_callback = ModelSaveCallback(gen.model, disc.model, model_path)
    tensorboard_callback = TensorBoard(log_dir=tensor_board_path)

    # create optimizers
    generator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    # compile and train
    wgangp = WGAN(discriminator=disc.model, generator=gen.model, classifier=classifier, input_shape=shape)

    wgangp.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
    )

    wgangp.fit(
        DataGeneratorGAN(train, num_classes, BATCH_SIZE),
        epochs=EPOCHS,
        initial_epoch=0,
        callbacks=[model_callback, tensorboard_callback, image_callback],
    )

    gen.model.save("generated/models/final_gen.keras")
    disc.model.save("generated/models/final_disc.keras")

    print("Training complete")
