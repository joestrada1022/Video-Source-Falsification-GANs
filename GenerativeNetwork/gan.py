import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import TensorBoard, Callback  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

from generator import Generator
from discriminator import Discriminator

import os


class GAN():
    def __init__(
        self,
        input_shape,
        model_files_path,
        tensorflow_files_path,
    ):

        self.input_shape = input_shape
        self.model = None
        self.model_name = None
        
        self.global_save_model_dir = self.__generate_model_path(model_files_path)
        self.global_tensorboard_dir = self.__generate_tensor_path(tensorflow_files_path)

    def __generate_model_name(self):
        model_name = f"GAN"

        return model_name

    def __generate_model_path(self, model_files_path):
        return model_files_path

    def __generate_tensor_path(self, tensorflow_files_path):
        return tensorflow_files_path

    def create_model(self, generator, discriminator, model_name=None):
        discriminator.trainable = False
        
        gen_input = layers.Input(shape=self.input_shape)
        labels = tf.repeat(tf.one_hot([0, 1], 2), repeats=1080, axis=0)
        generated_image = generator([gen_input, labels])

        disc_output = discriminator(generated_image)

        model = Model(gen_input, disc_output) 

        opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics='accuracy')
        
        model_name = self.__generate_model_name()

        self.model_name = model_name
        self.model = model

        return model

    def train(self, train_ds, val_ds_test, num_classes):
        pass

    def get_tensorboard_path(self):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save TensorBoard log-files.")

        # Create directory if not exists
        path = os.path.join(self.global_tensorboard_dir, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def print_model_summary(self):
        if self.model is None:
            print("Can't print model summary, self.model is None!")
        else:
            print(f"\nSummary of model:\n{self.model.summary()}")

    def get_save_model_path_acc(self, file_name):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save checkpoints.")

        # Create directory if not exists
        path = os.path.join(self.global_save_model_dir, self.model_name, "acc")
        if not os.path.exists(path):
            os.makedirs(path)

        # Append file name and return
        return os.path.join(path, file_name)

    def get_save_model_path_loss(self, file_name):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save checkpoints.")

        # Create directory if not exists
        path = os.path.join(self.global_save_model_dir, self.model_name, "loss")
        if not os.path.exists(path):
            os.makedirs(path)

        # Append file name and return
        return os.path.join(path, file_name)
        
    def get_callbacks(self):
        default_file_name = "fm-e{epoch:05d}.keras"
        save_model_path_acc = self.get_save_model_path_acc(default_file_name)
        save_model_path_loss = self.get_save_model_path_loss(default_file_name)

        save_model_acc = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path_acc,
                                                            monitor='val_accuracy',
                                                            save_best_only=True,
                                                            verbose=1,
                                                            save_weights_only=False,
                                                            save_freq='epoch')

        save_model_val = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path_loss,
                                                            monitor='val_loss',
                                                            save_best_only=True,
                                                            verbose=1,
                                                            save_weights_only=False,
                                                            save_freq='epoch')                                                     
                                                 

        tensorboard_cb = TensorBoard(log_dir=self.get_tensorboard_path())

        return [save_model_acc, save_model_val, tensorboard_cb]