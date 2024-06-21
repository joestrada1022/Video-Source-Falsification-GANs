import tensorflow as tf

from generator import Generator
from discriminator import Discriminator
from datagenGAN import DataSetGeneratorGAN
from datagenGAN import DataGeneratorGAN
from wgan import WGAN, GANMonitor, ModelSaveCallback
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

generator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)


def discriminator_loss(real_images, fake_images):
    real_loss = tf.reduce_mean(real_images)
    fake_loss = tf.reduce_mean(fake_images)
    return fake_loss - real_loss


def generator_loss(fake_image):
    return -tf.reduce_mean(fake_image)


EPOCHS = 25
BATCH_SIZE = 12

data_path = "data/frames"

dataset_maker = DataSetGeneratorGAN(data_path)

num_classes = len(dataset_maker.get_class_names())

shape = (1080 // 2, 1920 // 2, 3)
gen = Generator(shape, num_classes)
gen.create_model()
gen.print_model_summary()


disc = Discriminator(shape, num_classes)
disc.create_model()
disc.print_model_summary()

train = dataset_maker.create_dataset()
print(f"Train dataset contains {len(train)} samples")

image_callback = GANMonitor()
tensorboard_callback = TensorBoard(log_dir="generated/tensors")
# callback to save best model every epoch based on best loss

model_callback = ModelSaveCallback(gen.model, disc.model, "generated/models/")

wgangp = WGAN(discriminator=disc.model, generator=gen.model, input_shape=shape)

wgangp.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    d_loss_fn=discriminator_loss,
    g_loss_fn=generator_loss,
)

wgangp.fit(
    DataGeneratorGAN(train, num_classes, BATCH_SIZE),
    epochs=EPOCHS,
    initial_epoch=0,
    callbacks=[model_callback, tensorboard_callback, image_callback],
)

gen.model.save("generated/models/final_gen.keras")
disc.model.save("generated/models/final_disc.keras")
