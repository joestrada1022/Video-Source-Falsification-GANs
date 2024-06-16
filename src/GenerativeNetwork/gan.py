import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import TensorBoard, Callback  # type: ignore
from tensorflow.keras import layers, optimizers, metrics, utils  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

from generator import Generator
from discriminator import Discriminator
from datagenGAN import DataGeneratorGAN, DataSetGeneratorGAN
from utils import apply_cfa

import os
   
def get_callbacks(save_model_path_acc, save_model_path_loss, tensorboard_path):
    default_file_name = "fm-e{epoch:05d}.keras"
    save_model_path_acc = os.path.join(save_model_path_acc, default_file_name)
    save_model_path_loss = os.path.join(save_model_path_loss, default_file_name)

    save_model_acc = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path_acc,
                                                        monitor='gen_loss',
                                                        save_best_only=True,
                                                        verbose=1,
                                                        save_weights_only=False,
                                                        save_freq='epoch')

    save_model_val = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path_loss,
                                                        monitor='disc_loss',
                                                        save_best_only=True,
                                                        verbose=1,
                                                        save_weights_only=False,
                                                        save_freq='epoch')                                                     
                                                

    tensorboard_cb = TensorBoard(log_dir=tensorboard_path)

    return [save_model_acc, save_model_val, tensorboard_cb]

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

train_loss_metric = metrics.Mean(name='train_loss')

data_path = "/home/cslfiu/dev/cnn_vscf/frames"

dataset_maker = DataSetGeneratorGAN(data_path)

num_classes = len(dataset_maker.get_class_names())

train = dataset_maker.create_dataset()
print(f'Train dataset contains {len(train)} samples')


datagen = DataGeneratorGAN(train, num_classes=num_classes, batch_size=5)

# Get callbacks
callbacks = get_callbacks(save_model_path_acc="/home/cslfiu/dev/cnn_vscf/models/acc",
                            save_model_path_loss="/home/cslfiu/dev/cnn_vscf/models/loss",
                            tensorboard_path="/home/cslfiu/dev/cnn_vscf/tensorboard")

# Training loop
for epoch in range(10):
    print(f"Epoch {epoch+1}")
    for i, (frames_batch, labels_batch) in enumerate(datagen):
        tf.print(frames_batch.shape, labels_batch.shape, sep='\t')
        data = (frames_batch, labels_batch)
        gen_loss, disc_loss = train_step(data, g_opt, d_opt, gen, disc)
        train_loss_metric(gen_loss)
        train_loss_metric(disc_loss)
        if i % 10 == 0:
            print(f"Batch {i+1}: Gen Loss: {gen_loss}, Disc Loss: {disc_loss}")
    print(f"Epoch {epoch+1}: Gen Loss: {train_loss_metric.result()}")
    print(f"Epoch {epoch+1}: Disc Loss: {train_loss_metric.result()}")
    train_loss_metric.reset_states()
    # Save the model
    gen.model.save(f"/home/cslfiu/dev/cnn_vscf/models/gen-{epoch}.h5")
    disc.model.save(f"/home/cslfiu/dev/cnn_vscf/models/disc-{epoch}.h5")