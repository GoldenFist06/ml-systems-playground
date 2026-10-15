# Softmax Multi-Class Classifier from Scratch in Rust

A lightweight, zero-dependency Multi-Class Classifier built from scratch in Rust, utilizing the Softmax activation function and Cross-Entropy gradient updates.

## Overview
* **Architecture:** Multi-class classification model using raw logits mapped via Softmax.
* **Activation:** Softmax function with numerical stability (max-value subtraction to prevent overflow).
* **Optimization:** Gradient Descent with categorical error correction.

## Structure
* `src/classifier.rs`: Contains the `SoftmaxClassifier` struct, weights/biases initialization, forward pass, and training logic.
* `src/main.rs`: Execution setup, synthetic dataset iteration, and probability logging.

## Usage
Run the training loop and evaluation:
```bash
cargo run

ResultsAfter training for 5,000 epochs, the classifier successfully concentrates probabilities ( ≈ 99% )  on the correct target class for each input sample.
