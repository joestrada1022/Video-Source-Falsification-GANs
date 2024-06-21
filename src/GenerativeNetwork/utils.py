import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from glob import glob
import random
from matplotlib import pyplot as plt

def apply_cfa(image: np.ndarray) -> np.ndarray:
    """Applies a Bayer filter to an image

    Args:
        image (np.ndarray): Takes in an OpenCV image in BGR format

    Returns:
        _type_: OpenCV Image of the same dimensions also in BGR format with the Bayer filter applied
    """
    # resize before interpolation to keep dimensions the same
    image = cv2.resize(image, None, fx=3, fy=3)
    # interpolate to get the 'cubey' effect with bayer filter
    image = cv2.resize(image, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_NEAREST)
    # G R B G bayer pattern
    mask_r = np.zeros_like(image)
    mask_g = np.zeros_like(image)
    mask_b = np.zeros_like(image)
    mask_r[::2, 1::2, 0] = 1  # Red channel mask
    mask_g[::2, ::2, 1] = 1   # Green channel mask (even rows, even columns)
    mask_g[1::2, 1::2, 1] = 1 # Green channel mask (odd rows, odd columns)
    mask_b[1::2, ::2, 2] = 1  # Blue channel mask

    # Apply masks to original image
    cfa_image = np.zeros_like(image)
    cfa_image += mask_r * image
    cfa_image += mask_g * image
    cfa_image += mask_b * image
    
    return cfa_image

def display_samples(model_path, image_path=None, save_path=None):
    if image_path is None:
        image_path = random.choice(glob("data/frames/**/Validation/**/*.jpg"))
        print(f"Using random image: {image_path}")
    if type(model_path) == str:
        model = load_model(model_path)
    else:
        model = model_path
    img = cv2.imread(image_path)
    orig = img.copy()
    flip = False
    if img.shape[0] == 1920 and img.shape[1] == 1080:
        img = np.transpose(img, (1, 0, 2))
        flip = True
    height, width = img.shape[:2]
    img = cv2.resize(img, (width//2, height//2))
    img = (img - 127.5) / 127.5
    img = np.expand_dims(img, axis=0)

    output = model.predict(img)
    output = (output * 127.5) + 127.5
    output = output[0]
    if flip:
        output = np.transpose(output, (1, 0, 2))
    output = output.astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    # display original and output images side by side matplotlib
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(orig)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(output)
    axes[1].set_title("Generated Image")
    axes[1].axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return orig, output

