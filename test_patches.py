from cnn_ensemble_vsi.dataset.patch_extractor_all_overlapping import (
    crop_image_into_four_quadrants,
    get_patches,
)
from cnn_ensemble_vsi.CNN_base_learners.cnn_network import ConstrainedLayer

import tensorflow as tf
import keras
from utils.helpers import display_samples
from utils.datagenGAN import DataGeneratorGAN, DataSetGeneratorGAN
import numpy as np
import time
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

generator = keras.models.load_model("generated/wgan/models/generator_epoch_99.keras")

custom_objects = {"ConstrainedLayer": ConstrainedLayer}

patches_classifier1 = keras.models.load_model("generated/patch_classifier/models/quadrant_1/CNN_base_learner-quadrant_1/acc/fm-e00024.keras", custom_objects=custom_objects)
patches_classifier2 = keras.models.load_model("generated/patch_classifier/models/quadrant_2/CNN_base_learner-quadrant_2/acc/fm-e00018.keras", custom_objects=custom_objects)
patches_classifier3 = keras.models.load_model("generated/patch_classifier/models/quadrant_3/CNN_base_learner-quadrant_3/acc/fm-e00021.keras", custom_objects=custom_objects)
patches_classifier4 = keras.models.load_model("generated/patch_classifier/models/quadrant_4/CNN_base_learner-quadrant_4/acc/fm-e00024.keras", custom_objects=custom_objects)


data_factory = DataSetGeneratorGAN("data/frames")
test_set = data_factory.create_dataset('Testing')
print(f"Test dataset contains {len(test_set)} samples")
num_classes = len(data_factory.get_class_names())

datagen = DataGeneratorGAN(test_set, num_classes=num_classes, batch_size=32)

predictions = []
true_labels = []

start = time.time()
for i, (frames_batch, labels_batch) in enumerate(datagen):
    generated = generator.predict(frames_batch, verbose=0)
    # extract patches for every image in batch
    for img in generated:
        img = np.float32(img) * 127.5 + 127.5
        quadrant_1, quadrant_2, quadrant_3, quadrant_4 = crop_image_into_four_quadrants(img)
        p1 = get_patches(img_data=quadrant_1, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
        p2 = get_patches(img_data=quadrant_2, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
        p3 = get_patches(img_data=quadrant_3, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
        p4 = get_patches(img_data=quadrant_4, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
        # rehape the patches
        p1 = np.array(p1).reshape(-1, 128, 128, 3)
        p2 = np.array(p2).reshape(-1, 128, 128, 3)
        p3 = np.array(p3).reshape(-1, 128, 128, 3)
        p4 = np.array(p4).reshape(-1, 128, 128, 3)
        # classify the patches
        pred1 = patches_classifier1.predict(p1, verbose=0)
        pred2 = patches_classifier2.predict(p2, verbose=0)
        pred3 = patches_classifier3.predict(p3, verbose=0)
        pred4 = patches_classifier4.predict(p4, verbose=0)
        # print(np.argmax(pred1, axis=1))
        # print(np.argmax(pred2, axis=1))
        # print(np.argmax(pred3, axis=1))
        # print(np.argmax(pred4, axis=1))
        # get majority vote prediction for entire image
        pred1 = np.argmax(np.sum(pred1, axis=0))
        pred2 = np.argmax(np.sum(pred2, axis=0))
        pred3 = np.argmax(np.sum(pred3, axis=0))
        pred4 = np.argmax(np.sum(pred4, axis=0))

        image_pred = np.argmax(np.bincount([pred1, pred2, pred3, pred4]))

        # append the predictions
        predictions.append(image_pred)
    labels = np.argmax(labels_batch, axis=1)
    true_labels.append(labels)
end = time.time()


# get accuracy
predictions = np.array(predictions)
true_labels = np.array(true_labels)
true_labels = np.reshape(true_labels, (-1))
accuracy = np.mean(predictions == true_labels)
print(f"Accuracy: {accuracy:.4f}")
print(f"Time Taken: {end - start}")


# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predictions)

# Calculate misclassification rates
misclassification_counts = np.sum(cm, axis=1) - np.diag(cm)
total_counts = np.sum(cm, axis=1)
misclassification_rates = misclassification_counts / total_counts

# Assuming class_names is already defined
class_names = data_factory.get_class_names()

# Create a DataFrame to store the results
df = pd.DataFrame({
    'Class': class_names,
    'Misclassification Rate': np.round(misclassification_rates, 4)
})

# Plot the table
fig, ax = plt.subplots(figsize=(8, 2))  # set size frame
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
# plt.title("Model 1")
# Save the table as an image
plt.savefig('generated/statistics/misclassification_rates_table.png', bbox_inches='tight', dpi=300)

# Display the table
plt.show()




# # generate an image
# generated_image, labels = display_samples(generator, 'data/frames/', show=False)[1:]

# # crop the image into four quadrants
# quadrant_1, quadrant_2, quadrant_3, quadrant_4 = crop_image_into_four_quadrants(generated_image)

# p1 = get_patches(img_data=quadrant_1, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
# p2 = get_patches(img_data=quadrant_2, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
# p3 = get_patches(img_data=quadrant_3, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
# p4 = get_patches(img_data=quadrant_4, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
# # rehape the patches
# p1 = np.array(p1).reshape(-1, 128, 128, 3)
# p2 = np.array(p2).reshape(-1, 128, 128, 3)
# p3 = np.array(p3).reshape(-1, 128, 128, 3)
# p4 = np.array(p4).reshape(-1, 128, 128, 3)

# # classify the patches
# pred1 = patches_classifier1.predict(p1)
# pred2 = patches_classifier2.predict(p2)
# pred3 = patches_classifier3.predict(p3)
# pred4 = patches_classifier4.predict(p4)

# # print the results
# class_names = ["Galaxi-A50", "IPhone", "Nokia"]
# print("Quadrant 1:")
# print(np.argmax(pred1, axis=1))
# print([class_names[i] for i in np.argmax(pred1, axis=1)])
# print("Quadrant 2:")
# print(np.argmax(pred2, axis=1))
# print([class_names[i] for i in np.argmax(pred2, axis=1)])
# print("Quadrant 3:")
# print(np.argmax(pred3, axis=1))
# print([class_names[i] for i in np.argmax(pred3, axis=1)])
# print("Quadrant 4:")
# print(np.argmax(pred4, axis=1))
# print([class_names[i] for i in np.argmax(pred4, axis=1)])


# # print real label
# print("Real label:")
# print(np.argmax(labels))
# print(class_names[np.argmax(labels)])