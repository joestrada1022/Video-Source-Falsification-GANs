import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from keras.models import load_model
from utils.datagenGAN import DataGeneratorGAN, DataSetGeneratorGAN
import matplotlib.pyplot as plt
import pandas as pd

CLASSIFIER_PATH = 'generated/models/classifier/final_class.keras'
GENERATOR_PATH = 'generated/wgan/models/generator_epoch_91.keras'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
data_path = 'data/frames'

classifier = load_model(CLASSIFIER_PATH)
generator = load_model(GENERATOR_PATH)

data_factory = DataSetGeneratorGAN(data_path)
test_set = data_factory.create_dataset('Testing')
print(f"Test dataset contains {len(test_set)} samples")
num_classes = len(data_factory.get_class_names())

datagen = DataGeneratorGAN(test_set, num_classes=num_classes, batch_size=12)

predictions = []
true_labels = []
for i, (frames_batch, labels_batch) in enumerate(datagen):
    generated = generator.predict(frames_batch, verbose=0)
    class_predictions = classifier.predict(generated, verbose=0)
    predictions.append(class_predictions)
    true_labels.append(labels_batch)
    
predictions = np.array(predictions)
true_labels = np.array(true_labels)
predictions = np.reshape(predictions, (-1, num_classes))
true_labels = np.reshape(true_labels, (-1, num_classes))

print(f'Predictions shape: {predictions.shape}, True labels shape: {true_labels.shape}')
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(true_labels, axis=1))
print(f"Classifier accuracy: {accuracy}")

# Calculate the predicted and true classes
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(true_labels, axis=1)

# Get class names
class_names = data_factory.get_class_names()

# Calculate misclassification rates
misclassifications = (predicted_classes != true_classes)
misclassification_counts = np.bincount(true_classes[misclassifications], minlength=len(class_names))
total_counts = np.bincount(true_classes, minlength=len(class_names))
misclassification_rates = misclassification_counts / total_counts

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

# Save the table as an image
plt.savefig('generated/statistics/misclassification_rates_table.png', bbox_inches='tight', dpi=300)

# Display the table
plt.show()