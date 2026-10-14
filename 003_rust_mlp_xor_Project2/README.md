# Multi-Layer Perceptron (MLP) for XOR Gate in Rust

A lightweight, zero-dependency Multi-Layer Perceptron (MLP) built from scratch in Rust. This implementation demonstrates how a hidden layer combined with a non-linear activation function solves the non-linearly separable XOR problem.

## Overview
* **Architecture:** 2 Input Nodes $\rightarrow$ 2 Hidden Nodes $\rightarrow$ 1 Output Node.
* **Activation:** Sigmoid function ($\sigma(x) = \frac{1}{1 + e^{-x}}$).
* **Optimization:** Backpropagation with Gradient Descent.

## Structure
* `src/mlp.rs`: Contains the `Mlp` struct, matrix operations, forward pass, and backpropagation logic.
* `src/main.rs`: Execution setup for dataset iteration and prediction logging.

## Usage
Run the training loop and testing evaluation:
```bash
cargo run

ResultsAfter 20,000 training epochs, the predictions settle close to exact binary target values:

[0, 0] \rightarrow ~0.0111 (Target: 0)
[0, 1] \rightarrow ~0.9884 (Target: 1)
[1, 0] \rightarrow ~0.9884 (Target: 1)
[1, 1] \rightarrow ~0.0143 (Target: 0)