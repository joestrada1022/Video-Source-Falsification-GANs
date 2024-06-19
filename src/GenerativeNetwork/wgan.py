import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
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

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

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
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)

                # calculate discriminator loss
                d_cost = self.d_loss_fn(
                    real_images=real_logits, fake_images=fake_logits
                )
                # calculate gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # add the gp to the discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # get the gradients with respect to the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)

            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        # train generator
        with tf.GradientTape() as tape:
            # generate fake images
            generated_images = self.generator(real_images, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)

        # get the gradients with respect to the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        return {"d_loss": d_loss, "g_loss": g_loss}


class GANMonitor(Callback):
    def __init__(
        self,
        num_img=3,
    ):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        for i in range(self.num_img):
            img = display_samples(
                self.model.generator, save_path=f"img_{i}_epoch_{epoch}.png"
            )
