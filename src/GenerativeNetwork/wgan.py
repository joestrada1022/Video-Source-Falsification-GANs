import tensorflow as tf
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.callbacks import Callback  # type: ignore
from tensorflow.keras import metrics  # type: ignore
from utils import display_samples
from glob import glob
import os
import random


class WGAN(Model):
    def __init__(
        self,
        discriminator,
        generator,
        classifier,
        input_shape,
        discriminator_extra_steps=3,
        gp_weight=10.0,
        cls_weight=.01,  # TODO: adjust value
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.classifier = classifier

        self.input_shape = input_shape
        self.d_steps = discriminator_extra_steps

        self.gp_weight = gp_weight
        self.cls_weight = cls_weight

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

    @property
    def metrics(self):
        return [self.d_loss, self.d_wass_loss, self.d_gp, self.g_loss, self.cls_loss, self.adv_loss]

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

    def classifier_loss(self, generated_images, target_labels):
        cls_predictions = self.classifier(generated_images, training=False)
        cls_loss = -tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                target_labels, cls_predictions["output_0"]
            )
        )

        return cls_loss
    
    def classifier_prepare(self, generated_images):
        generated_images = tf.image.resize(generated_images, (270 * 4, 480 * 4))
        img_height, img_width = 270 * 4, 480 * 4
        crop_height, crop_width = 480, 800
        generated_images = tf.image.crop_to_bounding_box(
            image=generated_images,
            offset_height=int(img_height / 2 - crop_height / 2),
            offset_width=int(img_width / 2 - crop_width / 2),
            target_height=crop_height,
            target_width=crop_width,
        )

        return generated_images


    # @tf.function # if training slow, turn this one
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
                fake_images = self.generator([real_images, real_labels], training=True)
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
            generated_images = self.generator([real_images, real_labels], training=True)
            gen_predictions = self.discriminator(generated_images, training=True)

            adv_loss = -tf.reduce_mean(gen_predictions)

            # prepare images for classifier
            preprocessed_images = self.classifier_prepare(generated_images)

            # calculate classification loss
            cls_loss = self.classifier_loss(preprocessed_images, real_labels)
            # add other losses to the generator loss
            g_loss = (cls_loss * self.cls_weight) + adv_loss

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

        return {m.name: m.result() for m in self.metrics}


class GANMonitor(Callback):
    """
    Callback for monitoring and saving generated images during training.
    Shows the progression of the same generated image after each epoch.
    """

    def __init__(self, data_path: str, save_path: str, num_img:int =None):
        device_paths = sorted(list(glob(f"{data_path}*")))
        if len(device_paths) == 0:
            raise ValueError(f"No devices found in {data_path}")
        
        self.num_img = num_img if num_img else len(device_paths)
        self.save_path = save_path
        self.data_path = data_path
        imgs = []

        for i in range(self.num_img):
            if i < len(device_paths):
                device_path = device_paths[i]
                image_paths = list(glob(f"{device_path}/Validation/**/*.jpg"))

            else:
                image_paths = list(glob(f"{self.data_path}**/Validation/**/*.jpg"))

            img = random.choice(image_paths)
            imgs.append(img)
        self.images = imgs
        print(f"Using images: {[os.path.basename(img) for img in self.images]}")



    def on_epoch_end(self, epoch: int, logs=None):
        """
        Callback function called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary containing the training metrics for the current epoch. Defaults to None.
        """
        for i in range(self.num_img):
            display_samples(
                model_path=self.model.generator,
                data_path=self.data_path,
                save_path=f"{self.save_path}/epoch_{epoch}_img{i}.png",
                image_path=self.images[i],
                show=False,
            )


class ModelSaveCallback(Callback):
    """
    Callback to save the generator and discriminator models at the end of each epoch.

    Args:
        generator (tf.keras.Model): The generator model.
        discriminator (tf.keras.Model): The discriminator model.
        save_path (str): The directory path to save the models.

    Methods:
        on_epoch_end(epoch, logs=None):
            Saves the generator and discriminator models at the end of each epoch.

    Example:
        generator = create_generator_model()
        discriminator = create_discriminator_model()
        save_path = "/path/to/save/models"
        callback = ModelSaveCallback(generator, discriminator, save_path)
        model.fit(x_train, y_train, callbacks=[callback])
    """

    def __init__(self, generator: Model, discriminator: Model, save_path: str):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        """
        Saves the generator and discriminator models at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Dictionary containing the training metrics for the current epoch.

        Returns:
            None
        """
        self.generator.save(f"{self.save_path}/generator_epoch_{epoch}.keras")
        self.discriminator.save(f"{self.save_path}/discriminator_epoch_{epoch}.keras")
