import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers import display_samples
import datetime, cv2
from keras.models import load_model
import numpy as np


save_path = None

image_path = 'data/frames/Galaxi-A50/Validation/Samsung-A50-2(299)/Samsung-A50-2(299).mp4_011.jpg'
image_path = None

display_samples("generated/wgan/models/generator_epoch_90.keras", image_path=image_path, save_path=save_path, data_path="data/frames/", folder='Testing')
