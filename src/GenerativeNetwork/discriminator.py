import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import TensorBoard  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

import os


class Discriminator:
    def __init__(self, input_shape, num_classes):
        self.model = None
        self.model_name = None
        self.input_width, self.input_height, self.input_channels = input_shape
        self.num_classes = num_classes

    def __generate_model_name(self):
        model_name = f"Discriminator"

        return model_name

    """
    According to DCGAN, "Directly applying batchnorm to all layers however, 
    resulted in sample oscillation and model instability. 
    This was avoided by not applying batchnorm to the generator 
    output layer and the discriminator input layer."
    """

    # no batch norm for WGAN discriminator
    def create_model(self, model_name=None):
        shape = (self.input_width, self.input_height, self.input_channels)
        disc_input = layers.Input(shape=shape)

        conv2d_1 = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(
            disc_input
        )
        # no batch norm for input layer of disc
        act1 = layers.LeakyReLU()(conv2d_1)
        dropout1 = layers.Dropout(0.25)(act1)

        conv2d_2 = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(
            dropout1
        )
        # batch_norm1 = layers.BatchNormalization()(conv2d_2)
        act2 = layers.LeakyReLU()(conv2d_2)
        dropout2 = layers.Dropout(0.25)(act2)

        conv2d_3 = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(
            dropout2
        )
        # batch_norm2 = layers.BatchNormalization()(conv2d_3)
        act3 = layers.LeakyReLU()(conv2d_3)
        dropout3 = layers.Dropout(0.25)(act3)

        conv2d_4 = layers.Conv2D(512, kernel_size=4, strides=2, padding="same")(
            dropout3
        )
        # batch_norm3 = layers.BatchNormalization()(conv2d_4)
        act4 = layers.LeakyReLU()(conv2d_4)
        dropout4 = layers.Dropout(0.25)(act4)

        conv2d_final = layers.Conv2D(
            1, kernel_size=4, strides=1, padding="valid"
        )(dropout4)
        act5 = layers.LeakyReLU()(conv2d_final)
        output_layer = layers.Flatten()(act5)

        model = Model(disc_input, output_layer)

        self.model_name = self.__generate_model_name()
        self.model = model

        return model

    def print_model_summary(self):
        if self.model is None:
            print("Can't print model summary, self.model is None!")
        else:
            print(f"\nSummary of model:\n{self.model.summary()}")
