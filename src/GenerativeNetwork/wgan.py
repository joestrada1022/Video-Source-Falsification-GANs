import tensorflow as tf
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.callbacks import Callback #type: ignore
from tensorflow.keras import metrics #type: ignore
from utils import display_samples


class WGAN(Model):
    def __init__(
        self,
        discriminator,
        generator,
        input_shape,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.input_shape = input_shape
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_wass_loss = metrics.Mean(name="d_wasserstein_loss")
        self.d_gp = metrics.Mean(name="d_gradient_penalty")
        self.g_loss = metrics.Mean(name="g_loss")
        self.d_loss = metrics.Mean(name="d_loss")

    @property
    def metrics(self):
        return [self.d_loss, self.d_wass_loss, self.d_gp, self.g_loss]

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

    # @tf.function # if training slow, turn this on
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
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
            generated_images = self.generator(real_images, training=True)
            gen_predictions = self.discriminator(generated_images, training=True)
            g_loss = -tf.reduce_mean(gen_predictions)

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        self.d_loss.update_state(d_loss)
        self.d_wass_loss.update_state(d_wass_loss)
        self.d_gp.update_state(d_gp)
        self.g_loss.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}


class GANMonitor(Callback):
    def __init__(
        self,
        num_img=3,
    ):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        for i in range(self.num_img):
            img = display_samples(
                self.model.generator,
                save_path=f"generated/images/img_{i}_epoch_{epoch}.png",
            )


class ModelSaveCallback(Callback):
    def __init__(self, generator, discriminator, save_path):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.generator.save(f"{self.save_path}/generator_epoch_{epoch}.keras")
        self.discriminator.save(f"{self.save_path}/discriminator_epoch_{epoch}.keras")
