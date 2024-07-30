import tensorflow as tf
import numpy as np
import keras
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.datagenGAN import DataSetGeneratorGAN, DataGeneratorGAN

def _center_crop(self, img):
        img = tf.convert_to_tensor(img.numpy())
        img_height, img_width, _ = img.get_shape().as_list()

        # Correcting image orientation
        if img_height > img_width:
            img = tf.image.rot90(img)
            img_height, img_width = img_width, img_height

        # Perform center crop
        crop_height, crop_width = self.img_height, self.img_width
        img = tf.image.crop_to_bounding_box(image=img,
                                            offset_height=int(img_height / 2 - crop_height / 2),
                                            offset_width=int(img_width / 2 - crop_width / 2),
                                            target_height=crop_height,
                                            target_width=crop_width)
        return img

classifier_path = 'generated/models/cgan/discriminator_epoch_13.keras'

classifier = keras.models.load_model(classifier_path)

# create dataset
dataset_maker = DataSetGeneratorGAN('data/frames')

train = dataset_maker.create_dataset('Training')
print(f"Train dataset contains {len(train)} samples")
num_classes = len(dataset_maker.get_class_names())

shape = (1080 // 3, 1920 // 3, 3)

datagen = DataGeneratorGAN(train, num_classes=num_classes, batch_size=24)

# Iterate over the data generator
for i, (frames_batch, labels_batch) in enumerate(datagen):
    img_height, img_width = frames_batch.shape[1:3]
    print(img_height, img_width, sep='\t')
    img_height, img_width = 360, 640
    generated_images = frames_batch
    if img_height > img_width:
        generated_images = tf.image.rot90(generated_images)
        img_height, img_width = img_width, img_height
    crop_height, crop_width = 360, 640
    generated_images = tf.image.crop_to_bounding_box(
        image=generated_images,
        offset_height=int(img_height / 2 - crop_height / 2),
        offset_width=int(img_width / 2 - crop_width / 2),
        target_height=crop_height,
        target_width=crop_width,
    )
    # generated_images = generated_images / 255
    cls_predictions = classifier(generated_images, training=False)
    # print true label and predicted label
    print(f"True labels: {labels_batch.argmax(axis=1)}")
    print(f"Pred labels: {cls_predictions.numpy().argmax(axis=1)}")
    accuracy = np.mean(labels_batch.argmax(axis=1) == cls_predictions.numpy().argmax(axis=1))
    print(f"Accuracy: {accuracy}")
    if i >= 2: 
        break