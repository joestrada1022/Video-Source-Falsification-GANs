import tensorflow as tf
import numpy as np
import keras
from datagenGAN import DataSetGeneratorGAN, DataGeneratorGAN

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

classifier_path = 'scd-videos/results/12_frames/mobile_net/models/MobileNet_12/fm-e00020.keras'

classifier = keras.models.load_model(classifier_path)

# create dataset
dataset_maker = DataSetGeneratorGAN('data/frames')

train = dataset_maker.create_dataset('Training')
print(f"Train dataset contains {len(train)} samples")
num_classes = len(dataset_maker.get_class_names())

shape = (1080 // 4, 1920 // 4, 3)

datagen = DataGeneratorGAN(train, num_classes=num_classes, batch_size=32)

# Iterate over the data generator
for i, (frames_batch, labels_batch) in enumerate(datagen):
    img_height, img_width = frames_batch.shape[1:3]
    print(img_height, img_width, sep='\t')
    generated_images = tf.image.resize(frames_batch, (270 * 4, 480 * 4))
    img_height, img_width = 270 * 4, 480 * 4
    if img_height > img_width:
        generated_images = tf.image.rot90(generated_images)
        img_height, img_width = img_width, img_height
    crop_height, crop_width = 480, 800
    generated_images = tf.image.crop_to_bounding_box(
        image=generated_images,
        offset_height=int(img_height / 2 - crop_height / 2),
        offset_width=int(img_width / 2 - crop_width / 2),
        target_height=crop_height,
        target_width=crop_width,
    )
    # generated_images = generated_images / 127.5 - 127.5
    cls_predictions = classifier(generated_images, training=False)
    # print true label and predicted label
    print(f"True labels: {labels_batch.argmax(axis=1)}")
    print(f"Pred labels: {cls_predictions.numpy().argmax(axis=1)}")
    if i >= 2: 
        break