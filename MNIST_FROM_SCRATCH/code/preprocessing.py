import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
import random

def preprocess_image(img):
    img = img.convert("L")                           # Grayscale
    # img = ImageOps.invert(img)                       # Invert
    img = img.resize((28, 28))                       # Ensure size
    img = np.array(img).astype(np.float32) / 255.0   # Normalize

    # Center the digit (optional but useful)
    cy, cx = ndimage.center_of_mass(img)
    shift_y, shift_x = np.array(img.shape) // 2 - [cy, cx]
    img = ndimage.shift(img, shift=[shift_y, shift_x], mode='constant', cval=0.0)

    return img.reshape(784)  # Flatten


def augment_image(img_array, num_augments=3):
    augmented = []

    for _ in range(num_augments):
        img = img_array.reshape(28, 28)

        # Random rotation
        angle = random.uniform(-10, 10)
        img_rotated = ndimage.rotate(img, angle, reshape=False, mode='constant', cval=0.0)

        # Random shift
        shift_x = random.uniform(-2, 2)
        shift_y = random.uniform(-2, 2)
        img_shifted = ndimage.shift(img_rotated, shift=(shift_y, shift_x), mode='constant', cval=0.0)

        # Optional: Add Gaussian noise
        noise = np.random.normal(0, 0.02, (28, 28))
        img_noisy = np.clip(img_shifted + noise, 0.0, 1.0)

        # Flatten and store
        augmented.append(img_noisy.reshape(784))

    return augmented
