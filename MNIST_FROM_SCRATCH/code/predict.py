import sys
import os
import numpy as np
from PIL import Image, ImageOps
from network import NeuralNetwork

def preprocess_image(path):
    img = Image.open(path).convert("L")
    # img = ImageOps.invert(img)  # Uncomment if needed
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    return img.reshape(1, 784)

def get_image_paths(input_paths):
    all_images = []
    for path in input_paths:
        if os.path.isdir(path):
            for file in os.listdir(path):
                full_path = os.path.join(path, file)
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    all_images.append(full_path)
        elif os.path.isfile(path) and path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            all_images.append(path)
    return all_images

def predict(paths, model):
    nn = NeuralNetwork()
    nn.load(model)

    images = get_image_paths(paths)
    if not images:
        print("❌ No valid images found.")
        return

    for path in sorted(images):
        print(f"\n🔍 {os.path.basename(path)}")
        x = preprocess_image(path)
        probs = nn.forward(x)

        for i, prob in enumerate(probs[0]):
            print(f" {i}: {prob*100:.2f}%")

        pred = np.argmax(probs)
        confidence = probs[0][pred] * 100
        print(f"👉 Predicted: {pred} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <model_path> <folder_or_images>")
    else:
        predict(sys.argv[2:], sys.argv[1])
