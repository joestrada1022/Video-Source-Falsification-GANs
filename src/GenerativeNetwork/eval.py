from utils import display_samples, apply_cfa
import datetime, cv2
from tensorflow.keras.models import load_model
import numpy as np

id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"generated/sample-{id}.jpg"
save_path = None

image_path = 'data/frames/Galaxi-A50/Validation/Samsung-A50-2(299)/Samsung-A50-2(299).mp4_011.jpg'
image_path = None

_, output = display_samples("generated/models/final_gen.keras", image_path=image_path, save_path=save_path)
# orig = cv2.resize(orig, None, fx=.5, fy=.5)
if output.shape[0] == 960 and output.shape[1] == 540:
        output = np.transpose(output, (1, 0, 2))
# process output to be able to be fed into the discriminator
output = (output - 127.5) / 127.5
output = np.expand_dims(output, axis=0)

discriminator = load_model("generated/models/final_disc.keras")

# get prediction from discriminator
prediction = discriminator.predict(output)
print(prediction)