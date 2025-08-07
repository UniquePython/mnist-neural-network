import numpy as np
import struct
import os
from PIL import Image
from preprocessing import preprocess_image, augment_image

def load_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
        return images / 255.0  # normalize to [0,1]


def load_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


def load_custom_data(folder="custom_data"):
    x_data = []
    y_data = []

    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            label = int(filename.split("_")[0])
            path = os.path.join(folder, filename)

            img = Image.open(path)
            processed_img = preprocess_image(img)
            x_data.append(processed_img)  # Flatten to 784
            y_data.append(label)
            
            # Data augmentation
            for aug in augment_image(processed_img):
                x_data.append(aug)
                y_data.append(label)

    x = np.array(x_data)
    y = np.array(y_data)
    return x, y