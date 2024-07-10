import tensorflow as tf
from keras.models import Model
from keras.callbacks import Callback
from keras import layers, metrics, losses
from utils.helpers import display_samples
from glob import glob
import os, random
import numpy as np


class CGAN(Model):
    def __init__(
        self,
        classifier,
        generator,
        input_shape,
        num_classes,
        embedding_dim=50,
    ):
        super().__init__()
        self.model_input_shape = input_shape
        self.generator = self.modify_generator(generator, num_classes, embedding_dim)
        self.classifier = classifier
        self.num_classes = num_classes


    def compile(self, g_optimizer, c_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.g_loss = metrics.Mean(name="g_loss")
        self.c_loss = metrics.Mean(name="c_loss")

    def get_classifier(self):
        return self.classifier
    
    def get_generator(self):
        return self.generator

    @property
    def metrics(self):
        return [self.g_loss, self.c_loss]

    def modify_generator(self, original_generator, num_classes, embedding_dim):
        image_input = layers.Input(shape=self.model_input_shape)
        label_input = layers.Input(shape=(1,), dtype="int32")

        # layer that converts one hot encoded labels to integers
        label_input_int = layers.Lambda(lambda x: tf.argmax(x, axis=1))(label_input)
        # embedding for labels
        label_embedding = layers.Embedding(num_classes, embedding_dim)(label_input_int)
        label_embedding = layers.Flatten()(label_embedding)
        # reshape label embedding to have same dimensions as image
        label_embedding = layers.Dense(
            self.model_input_shape[0] * self.model_input_shape[1] * self.model_input_shape[2]
        )(label_embedding)
        label_embedding = layers.Reshape(
            (self.model_input_shape[0], self.model_input_shape[1], 3), 
            name="label_reshape"
        )(label_embedding)

        # concatenate image and label embedding
        concatenated = layers.Concatenate(axis=-1)([image_input, label_embedding])

        # use original generator layers but with new input
        x = layers.Conv2D(3, 1, padding='same', activation='relu', name='reduce_label')(concatenated)
        for layer in original_generator.layers[1:]:  # skip original input layer
            x = layer(x)

        new_generator = Model([image_input, label_input], x)

        for layer in new_generator.layers:
            if layer.name in [l.name for l in original_generator.layers]:
                layer.set_weights(original_generator.get_layer(layer.name).get_weights())

        new_generator.summary()

        return new_generator

    # @tf.function  # if training slow, turn this one
    def train_step(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            real_images, real_labels = data
        else:
            raise ValueError("Expected data format: (images, labels)")
        
        # turn one hot encoded labels into integers
        # real_labels = tf.argmax(real_labels, axis=1)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as cls_tape:
            # generate images
            generated_images = self.generator([real_images, real_labels], training=True)

            real_output = self.classifier(real_images, training=True)
            fake_output = self.classifier(generated_images, training=True)

            # classifier loss
            cls_loss = tf.reduce_mean(
                losses.categorical_crossentropy(real_output, real_labels)
                + losses.categorical_crossentropy(fake_output, real_labels)
            )

            # generator loss
            g_loss = tf.reduce_mean(losses.categorical_crossentropy(fake_output, real_labels))

        g_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        c_gradients = cls_tape.gradient(cls_loss, self.classifier.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.c_optimizer.apply_gradients(zip(c_gradients, self.classifier.trainable_variables))

        self.g_loss.update_state(g_loss)
        self.c_loss.update_state(cls_loss)

        return {m.name: m.result() for m in self.metrics}
