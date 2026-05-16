# IA — Learning Initiative

This repository is a personal learning initiative to understand the foundations of artificial intelligence and machine learning from scratch.

## Goal

Build intuition around how a neural network works by implementing each component manually — no high-level ML framework, only NumPy.

## Content

### `python/neuron.py`

Core implementation of an artificial neuron (logistic regression):
- `model` — sigmoid activation: computes the probability output
- `cout` — binary cross-entropy loss
- `gradients` — computes dW and db
- `update` — gradient descent step
- `artificial_neuron` — training loop combining the above

### Notebooks

| Notebook | Dataset | Task |
|---|---|---|
| `first_neuron.ipynb` | Synthetic 2D blobs | Build the neuron from scratch, visualize the decision boundary |
| `plants.ipynb` | Synthetic 2D blobs | Classify plants as toxic (1) or safe (0) |
| `catsordog.ipynb` | HDF5 images 64×64 | Classify images as cat (0) or dog (1) |

All three notebooks apply the same neuron to progressively more concrete problems.

## What I learned

- Forward pass: linear combination → sigmoid activation
- Loss: binary cross-entropy
- Backward pass: analytical gradients for W and b
- Gradient descent update rule
- Image flattening and normalization for pixel input

## Stack

- Python 3
- NumPy
- Matplotlib
- scikit-learn (dataset generation + accuracy metric)
- h5py (image dataset loading)
