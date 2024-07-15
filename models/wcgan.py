import tensorflow as tf
from keras.models import Model
from keras import metrics, layers
from keras.losses import categorical_crossentropy
import numpy as np


class WCGAN(Model):
    def __init__(
        self,
        discriminator,
        generator,
        classifier,
        input_shape,
        num_classes,
        embedding_dim=50,
        discriminator_extra_steps=3,
        gp_weight=10.0,
        cls_weight=1.0,  # TODO: adjust weights
        adv_weight=1.0,
        perceptual_weight=1.0,
    ):
        super().__init__()
        self.model_input_shape = input_shape

        self.discriminator = discriminator
        self.discriminator.summary()
        self.generator = generator
        self.generator.summary()
        self.classifier = classifier

        self.d_steps = discriminator_extra_steps

        self.gp_weight = gp_weight  # gradient penalty weight (discriminator)

        # weights for the losses (generator)
        self.cls_weight = cls_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_wass_loss = metrics.Mean(name="d_wasserstein_loss")
        self.d_gp = metrics.Mean(name="d_gradient_penalty")
        self.g_loss = metrics.Mean(name="g_loss")
        self.d_loss = metrics.Mean(name="d_loss")
        self.cls_loss = metrics.Mean(name="cls_loss")
        self.adv_loss = metrics.Mean(name="adv_loss")
        self.p_loss = metrics.Mean(name="perceptual_loss")

    # def modify_generator(self, original_generator, num_classes, embedding_dim):
    #     image_input = layers.Input(shape=self.model_input_shape)
    #     label_input = layers.Input(shape=(1,), dtype="int32")

    #     # layer that converts one hot encoded labels to integers
    #     label_input_int = layers.Lambda(lambda x: tf.argmax(x, axis=1))(label_input)
    #     # embedding for labels
    #     label_embedding = layers.Embedding(num_classes, embedding_dim)(label_input_int)
    #     label_embedding = layers.Flatten()(label_embedding)
    #     # reshape label embedding to have same dimensions as image
    #     label_embedding = layers.Dense(
    #         self.model_input_shape[0] * self.model_input_shape[1] * self.model_input_shape[2]
    #     )(label_embedding)
    #     label_embedding = layers.Reshape(
    #         (self.model_input_shape[0], self.model_input_shape[1], 3),
    #         name="label_reshape",
    #     )(label_embedding)

    #     # concatenate image and label embedding
    #     concatenated = layers.Concatenate(axis=-1)([image_input, label_embedding])

    #     # use original generator layers but with new input
    #     x = concatenated
    #     for layer in original_generator.layers[1:]:  # skip original input layer
    #         x = layer(x)

    #     new_generator = Model([image_input, label_input], x)



    #     new_generator.summary()

    #     return new_generator

    @property
    def metrics(self):
        return [
            self.d_loss,
            self.d_wass_loss,
            self.d_gp,
            self.g_loss,
            self.cls_loss,
            self.adv_loss,
            self.p_loss,
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # get discriminator output for the interpolated image
            pred = self.discriminator(interpolated, training=True)

        # calculate gradients with respect to the interpolated image
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # compute the norm of the gradients (euclidean norm. do research on this)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def classifier_loss(self, generated_images, labels):
        cls_predictions = self.classifier(generated_images, training=False)
        label = tf.argmax(labels, axis=1)
        prediction = tf.argmax(cls_predictions, axis=1)
        if prediction.shape != label.shape:
            raise ValueError("Length of predictions and labels do not match")
        
        # if true, if false (applies to each batch)
        factors = tf.where(tf.equal(prediction, label), -1.0, 1.0)
        
        cls_loss = categorical_crossentropy(labels, cls_predictions)

        cls_loss = tf.reduce_mean(tf.multiply(cls_loss, factors))

        return cls_loss

    def perceptual_loss(self, img1, img2):
        return tf.reduce_mean(tf.abs(img1 - img2))

    @tf.function  # if training slow, turn this one
    def train_step(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            real_images, real_labels = data
        else:
            raise ValueError("Expected data format: (images, labels)")
        # get batch size
        batch_size = tf.shape(real_images)[0]

        # train discriminator
        for i in range(self.d_steps):
            with tf.GradientTape() as tape:
                # generate fake images
                fake_images = self.generator(real_images, training=True)
                # get discriminator output for real and fake images
                fake_predictions = self.discriminator(fake_images, training=True)
                real_Predictions = self.discriminator(real_images, training=True)

                # calculate wasserstein loss
                d_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                    real_Predictions
                )
                d_gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_wass_loss + d_gp * self.gp_weight

            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables)
            )

        # train generator
        with tf.GradientTape() as tape:
            # calculate adversarial loss
            generated_images = self.generator(real_images, training=True)
            gen_predictions = self.discriminator(generated_images, training=True)

            adv_loss = -tf.reduce_mean(gen_predictions)

            # calculate classification loss
            cls_loss = self.classifier_loss(generated_images, real_labels)

            # calculate perceptual loss
            perceptual_loss = self.perceptual_loss(real_images, generated_images)

            # add other losses to the generator loss
            g_loss = (
                (cls_loss * self.cls_weight)
                + (adv_loss * self.adv_weight)
                + (perceptual_loss * self.perceptual_weight)
            )

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        self.d_loss.update_state(d_loss)
        self.d_wass_loss.update_state(d_wass_loss)
        self.d_gp.update_state(d_gp)
        self.g_loss.update_state(g_loss)
        self.cls_loss.update_state(cls_loss)
        self.adv_loss.update_state(adv_loss)
        self.p_loss.update_state(perceptual_loss)

        return {m.name: m.result() for m in self.metrics}
