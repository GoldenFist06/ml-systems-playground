# 🚀 Rust-Powered ML Core Engine

A high-performance, lightweight machine learning core and neural network engine built from scratch. It leverages **Rust** for the heavy mathematical computation core (ensuring memory safety, zero-cost abstractions, and cache-friendly vector layouts) and exposes a clean, intuitive **Python** interface via **PyO3** and **Maturin**.

---

## ⚡ Architecture & Design

```text
[ Python Interface (SimpleMLP / Training Loop) ]
                       │  (FFI via PyO3)
                       ▼
[ Rust Computation Core (PyMatrix & Tensors)  ]
  ├── Contiguous memory layout (Vec<f32>)
  ├── Native Matrix Multiplication (MatMul)
  ├── Element-wise Operations & Hadamard Product
  └── ReLU & Backpropagation Gradients Engine


  🛠️ Tech Stack
Core: Rust (Safety, Performance, Native SIMD-friendly structures)

Interface: Python (Dynamic model definition and high-level wrappers)

FFI Bridge: PyO3 & Maturin (Lightning-fast interoperability without overhead)

📊 Core Features
Memory-Safe Tensors (PyMatrix): Contiguous Vec<f32> layout maximizing CPU cache efficiency.

Complete Math Kernel: Addition, Subtraction, Matrix Multiplication (matmul), Transpose, Hadamard Product, and Scaling.

Deep Learning Primitives: Built-in ReLU activation, its backward derivative (relu_backward), and Xavier/Glorot weight initialization.

End-to-End Training Loop: Fully functional Forward and Backward passes (Backpropagation) optimized for fast gradient descent.

🚀 Getting Started & Installation
Prerequisites
Make sure you have Rust (Cargo) and Python installed, along with maturin:

Bash
pip install maturin


Build & Install in Development Mode
Clone the repository and compile the Rust core into a local Python package:

Bash
maturin develop

Running the Training Simulation:

Bash
python src/network.py


Running Performance Benchmarks:

Bash
python src/benchmark.py


📈 Performance Benchmarks
Matrix Multiplication (500x500): Executed in milliseconds natively.

Zero-Overhead Memory Migration: Seamless conversion between Python lists and Rust vectors via PyMatrix.from_list.





License: This project is licensed under the PolyForm Noncommercial License 1.0.0. Free for personal and educational use.
 Any commercial, business, or enterprise use requires an explicit commercial license from the author.