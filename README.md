# Neural Network from Scratch 

A lightweight, 2-layer Multilayer Perceptron (MLP) built entirely from scratch using only Python and NumPy. This project was developed as a hands-on exploration of deep learning fundamentals for the CAP 4105 (Machine Learning) course at Florida International University.

By avoiding high-level frameworks like PyTorch or TensorFlow, this project explicitly demonstrates the underlying mathematics of neural networks, including manual backpropagation, chain-rule calculus, and gradient descent optimization.

## Features

* **Zero-Dependency Architecture**: Built solely with standard Python and `numpy` for matrix/array operations.
* **Custom Forward Pass**: Implements a 2-input, 2-hidden, 1-output network architecture.
* **Manual Backpropagation**: Explicit mathematical implementation of the chain rule to compute gradients for all 9 network parameters.
* **Binary Cross-Entropy Loss**: Custom loss function with built-in protections against log(0) domain errors.
* **Gradient Descent Optimization**: Full training loop with iterative weight updates.

## Dataset

The network is trained on a synthetic dataset representing 10 students. 
* **Features (X)**: Two continuous features (e.g., hours studied, previous grades).
* **Target (y)**: Binary classification predicting whether the student passes (1) or fails (0).

## Network Architecture

The Multilayer Perceptron is structured as a **2-2-1** network:
1.  **Input Layer**: 2 features
2.  **Hidden Layer**: 2 neurons with **Sigmoid** activation
3.  **Output Layer**: 1 neuron with **Sigmoid** activation

### The Mathematics (Backpropagation)
The backward pass manually computes the error gradients with respect to each weight and bias:
* **Output Error**: `dL/dzo = y_hat - y`
* **Chain Rule through Sigmoid**: `sigma'(z) = sigma(z) * (1 - sigma(z))`
