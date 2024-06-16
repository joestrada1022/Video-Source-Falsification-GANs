from datagenGAN import DataGeneratorGAN, DataSetGeneratorGAN
import matplotlib.pyplot as plt

data_path = "/home/cslfiu/dev/cnn_vscf/frames"

dataset_maker = DataSetGeneratorGAN(data_path)

num_classes = len(dataset_maker.get_class_names())

train = dataset_maker.create_train_dataset()
print(f'Train dataset contains {len(train)} samples')


datagen = DataGeneratorGAN(train, num_classes=num_classes)

# Iterate over the data generator
for i, (frames_batch, labels_batch) in enumerate(datagen):
    print(f"Batch {i+1}:")
    print(f"Frames batch shape: {frames_batch.shape}")
    print(f"Labels batch shape: {labels_batch.shape}")
    if i >= 10:  # stop after 10 batches
        break
    # display the most recent image
plt.imshow(frames_batch[-1])
plt.show()