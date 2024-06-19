import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tensorflow.keras.callbacks import TensorBoard, Callback  # type: ignore
from tensorflow.keras import layers, optimizers, metrics, utils  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

from generator import Generator
from discriminator import Discriminator
from datagenGAN import DataGeneratorGAN, DataSetGeneratorGAN
import os, datetime


# LD = log D(I) + log(1 − D(G(I')))
def d_loss(real_output, fake_output):
    real_loss = tf.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = tf.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# La = log(1 − D(G(I'))),
def g_loss(fake_output):
    return tf.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)


# image to image translation
@tf.function
def train_step(data, g_opt, d_opt, gen, disc):
    images, labels = data
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        fake_images = gen.model(images, training=True)

        # Calculate discriminator loss
        real_output = disc.model(images, training=True)
        fake_output = disc.model(fake_images, training=True)
        disc_loss = d_loss(real_output, fake_output)

        # Calculate generator loss
        gen_loss = g_loss(fake_output)

    # Calculate gradients
    gen_gradients = gen_tape.gradient(gen_loss, gen.model.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, disc.model.trainable_variables)

    # Update generator and discriminator weights
    g_opt.apply_gradients(zip(gen_gradients, gen.model.trainable_variables))
    d_opt.apply_gradients(zip(disc_gradients, disc.model.trainable_variables))

    return gen_loss, disc_loss


shape = (1080, 1920, 3)
gen = Generator(shape, 4)
gen.create_model()
gen.print_model_summary()
g_opt = optimizers.Adam(1e-4)


disc = Discriminator(shape, 4)
disc.create_model()
disc.print_model_summary()
d_opt = optimizers.Adam(1e-4)

train_loss_metric = metrics.Mean(name="train_loss")

data_path = "data/frames"

dataset_maker = DataSetGeneratorGAN(data_path)

num_classes = len(dataset_maker.get_class_names())

train = dataset_maker.create_dataset()
print(f"Train dataset contains {len(train)} samples")


datagen = DataGeneratorGAN(train, num_classes=num_classes, batch_size=2)

current_time = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
train_log_dir = "generated/tensors/gan" + current_time + "/"
os.makedirs(os.path.join(train_log_dir, "gen"))
os.makedirs(os.path.join(train_log_dir, "disc"))

model_output_dir = "generated/models/gan" + current_time + "/"
os.makedirs(model_output_dir)

gen_summary_writer = tf.summary.create_file_writer(train_log_dir + 'gen')
disc_summary_writer = tf.summary.create_file_writer(train_log_dir + 'disc')


EPOCHS = 25

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")

    # get total number of batches
    num_batches = len(datagen)

    # initialize progress bar
    progress = tqdm(total=num_batches, desc="Epoch {}".format(epoch + 1))

    for i, (frames_batch, labels_batch) in enumerate(datagen):
        # tf.print(frames_batch.shape, labels_batch.shape, sep='\t')
        data = (frames_batch, labels_batch)
        gen_loss, disc_loss = train_step(data, g_opt, d_opt, gen, disc)
        train_loss_metric(gen_loss)
        train_loss_metric(disc_loss)
        if i % 10 == 0:
            progress.set_description(
                f"Batch {i+1}: Gen Loss: {gen_loss}, Disc Loss: {disc_loss}"
            )

        # if i % 100 == 0:
            # with gen_summary_writer.as_default():
            #     tf.summary.scalar("batch_loss", tf.reduce_mean(gen_loss), step=i)
            # with disc_summary_writer.as_default():
            #     tf.summary.scalar("batch_loss", tf.reduce_mean(disc_loss), step=i)


        # update progress bar
        progress.update(1)

    print(f"Epoch {epoch+1}: Gen Loss: {gen_loss}")
    print(f"Epoch {epoch+1}: Disc Loss: {disc_loss}")
    print(f"Epoch {epoch+1}: Train Loss: {train_loss_metric.result()}")

    progress.close()

    # Save the model
    print(f"Saving model at epoch {epoch+1}")
    gen.model.save(f"{model_output_dir}/gen-{epoch+1}.keras")
    disc.model.save(f"{model_output_dir}/disc-{epoch+1}.keras")

    with gen_summary_writer.as_default():
        tf.summary.scalar("batch_loss", tf.reduce_mean(gen_loss), step=epoch)
    with disc_summary_writer.as_default():
        tf.summary.scalar("batch_loss", tf.reduce_mean(disc_loss), step=epoch)    

    train_loss_metric.reset_state()
