import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers import display_samples
import datetime, cv2
from keras.models import load_model
import numpy as np


def get_classifier_accuracy(
    generator_model_path, classifier_model_path, data_path, iter=32
):
    predictions = []
    true_labels = []
    outputs = []
    for i in range(iter):
        # Load the generator model and generate images
        _, output, true_label = display_samples(
            generator_model_path, folder="Testing", data_path=data_path, show=False
        )

        # Preprocess the generated images
        if output.shape[0] == 640 and output.shape[1] == 360:
            output = np.transpose(output, (1, 0, 2))
        outputs.append(output)
        true_labels.append(np.argmax(true_label))
    # load classifier
    classifier = load_model(classifier_model_path)
    output = np.array(outputs)
    output = (output - 127.5) / 127.5
    # output = np.expand_dims(output, axis=0)

    # get prediction from classifier
    predictions = classifier.predict(output)

    true_labels = np.array(true_labels)

    # Calculate accuracy
    accuracy = np.mean(np.argmax(predictions, axis=1) == true_labels)
    return accuracy


# id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# save_path = f"generated/sample-{id}.jpg"
# save_path = None

# image_path = 'data/frames/Nokia/Validation/Nokia-7dot1-2(36)/Nokia-7dot1-2(36).mp4_002.jpg'
# image_path = None

# orig, output, _ = display_samples("generated/models/working_untargeted/generator_epoch_29.keras", image_path=image_path, save_path=save_path, data_path="data/frames/", show=True)
# # orig = cv2.resize(orig, None, fx=.5, fy=.5)
# if output.shape[0] == 640 and output.shape[1] == 360:
#         output = np.transpose(output, (1, 0, 2))
# # process output to be able to be fed into the discriminator
# output = (output - 127.5) / 127.5
# output = np.expand_dims(output, axis=0)

# discriminator = load_model("generated/models/classifier/final_class.keras")

# # get prediction from discriminator
# prediction = discriminator.predict(output)
# print(np.argmax(prediction))

# test function
if __name__ == "__main__":
    generator_model_path = (
        "generated/models/working_untargeted/generator_epoch_29.keras"
    )
    classifier_model_path = "generated/models/classifier/final_class.keras"
    data_path = "data/frames/"
    accuracy = get_classifier_accuracy(
        generator_model_path, classifier_model_path, data_path
    )
    print(f"Classifier accuracy: {accuracy:.2f}%")
