# 🧠 MNIST Digit Recognizer — From Scratch in NumPy

This project is a handwritten digit recognizer built entirely **from scratch using NumPy** — no external machine learning libraries like TensorFlow, PyTorch, or Keras were used. It replicates a basic neural network with forward/backward propagation, softmax, ReLU, and cross-entropy loss.

---

## ✨ Features

- Fully custom neural network with:
  - Arbitrary number of hidden layers
  - Xavier/He initialization
  - ReLU + Softmax activations
  - Cross-entropy loss + Gradient descent
- Dropout regularization support
- Custom data loading (from folders of `.png`)
- Training on custom handwritten digits
- Test accuracy tracking
- Exportable and loadable models (`.npz`)
- CLI prediction tool that works on any image

---

## 🗂️ Project Structure

```
MNIST_FROM_SCRATCH/
├── code/ # All Python source code
│ ├── network.py # Core neural network logic
│ ├── predict.py # CLI tool for predictions
│ ├── data_loader.py # Loads MNIST/custom data
│ ├── nn_math.py # Activation functions & loss
│ ├── main.py # Training script
│ ├── preprocessing.py # Preprocesses custom data to match MNIST preprocessing
├── custom_data/ # Your own 28×28 black-bg white-digit PNGs
├── data/ # Raw MNIST files (idx format)
├── models/ # Trained .npz model files
├── test_data/ # Digits to test the model
└── README.md
```

---

# Predicting images

To classify images:
```bash
python predict.py <model_path> <image(s)_or_folder>
```

I suggest using the `augmented_more_epochs.npz` model from the `models/` folder as it is the best one. 
