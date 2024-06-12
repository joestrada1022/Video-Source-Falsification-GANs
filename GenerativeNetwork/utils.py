import cv2
import numpy as np

def apply_cfa(image):
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