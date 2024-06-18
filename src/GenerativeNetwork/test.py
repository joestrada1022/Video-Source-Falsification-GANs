from datagenGAN import DataGeneratorGAN, DataSetGeneratorGAN
import cv2
import datetime

data_path = "data/frames"

dataset_maker = DataSetGeneratorGAN(data_path)

num_classes = len(dataset_maker.get_class_names())

train = dataset_maker.create_dataset()
print(f'Train dataset contains {len(train)} samples')


datagen = DataGeneratorGAN(train, num_classes=num_classes)

# Iterate over the data generator
for i, (frames_batch, labels_batch) in enumerate(datagen):
    print(f"Batch {i+1}:")
    print(f"Frames batch shape: {frames_batch.shape}")
    print(f"Labels batch shape: {labels_batch.shape}")
    if i >= 2:  # stop after 10 batches
        break
    # display the most recent image
img = frames_batch[-1]
img = (img * 127.5) + 127.5

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# make custom id for image to save
id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


cv2.imwrite(f"generated/batch_testing/test{id}.jpg", img)