# 🚀 ML Core Engine: High-Performance GPU-Accelerated Machine Learning Framework

An ultra-optimized, GPU-accelerated machine learning core engine built from scratch using **Rust** for blazing-fast hardware execution and **Python** for high-level orchestration, bound via **PyO3/FFI**. 

Designed to bypass the traditional overheads of deep learning interpreters, this engine achieves high-throughput matrix computations and memory-efficient kernel operations.

---

## ⚠️ Note on Project Lineage

> This repository is the heavily optimized, patched, and performance-upgraded evolution of the legacy `006_ml_core_engine_GPU-Performance.Project5` project, featuring architectural redesigns like Kernel Fusion and a Persistent GPU Context.

---

## 🏗️ Architectural Overview

Modern machine learning runtimes often suffer from latency overheads caused by frequent data transfers between host (CPU) and device (GPU) memory, as well as separate kernel execution passes for operations and activations. 

This project tackles these bottlenecks through two core innovations:
* **Persistent GPU Context Architecture:** Keeps the compute context alive and persistent, eliminating initialization latency across operations.
* **Kernel Fusion (Fused Operations):** Combines heavy compute-bound kernels (like Matrix Multiplication) with activation functions (ReLU) into a single unified GPU compute pass.

---

## 🛠️ Key Architectural Enhancements & Solved Bottlenecks

### 1. Eliminating the ReLU Memory Bottleneck (Kernel Fusion)
* **The Problem:** In traditional implementations, a matrix multiplication (`MatMul`) writes its massive intermediate results back to the global VRAM, and then a subsequent `ReLU` kernel reads those results back from VRAM, applies the threshold, and writes them back again. This creates a severe **Memory Bandwidth Bottleneck**.
* **The Solution:** We engineered a **Fused GPU Kernel** (`gpu_matmul(..., use_relu=True)`) that applies the ReLU activation *on-the-fly* inside the GPU registers/shared memory right as the dot-product elements are computed.
* **The Impact:** Zero intermediate VRAM write/read round-trips for activations, drastically slashing memory traffic and unlocking massive performance scaling under heavy loads.

### 2. Zero-Overhead Python-Rust Bridge
* Leveraging **PyO3**, the tensor matrices are managed directly in Rust's raw, safe memory allocations while exposing an intuitive, native Python API (`PyMatrix`). This removes interpreter bloat and ensures raw hardware-level speed.

---

## 📊 Performance Benchmarks & Stress Testing

Recent hardware stress tests conducted on massive matrix scales demonstrate the engine's true saturation capabilities and scaling efficiency:

| Matrix Size | Standard GPU MatMul | Fused MatMul + ReLU (New Core) | Performance (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **$500 \times 500$** | 0.063 s | 0.028 s | — |
| **$1000 \times 1000$** | 0.255 s | 0.254 s | 7.86 GFLOPS |
| **$2000 \times 2000$** | 1.820 s | 1.619 s | 9.88 GFLOPS |
| **$4000 \times 4000$** (Full Saturation) | 6.320 s | 0.879 s 🚀 | **145.49 GFLOPS** |

> **Key Takeaway from Stress Tests:** At scale ($4000 \times 4000$), the GPU achieves complete hardware saturation, and the Kernel Fusion architecture delivers a dramatic performance leap—processing 16 million elements per matrix in under **0.88 seconds**.

---

## ⚠️ Hardware Performance Disclaimer

> **Tested Environment:** The performance benchmarks and stress tests documented in this project (including the $4000 \times 4000$ hardware saturation test reaching **145.49 GFLOPS**) were executed and verified on an **NVIDIA GeForce RTX 4050 GPU**. 
> 
> * **Performance Scaling Note:** Hardware acceleration, throughput (GFLOPS), and execution times heavily depend on your specific GPU architecture, VRAM bandwidth, CUDA core count, and system specifications. Results may scale higher on more powerful hardware or vary on lower-end setups.

---

## 📂 Project Structure

```text
006_ml_core_engine_GPU-Performance.Project5/
│
├── src/
│   ├── benchmark.py       # Standard and Fused kernel performance checks
│   ├── stress_test.py     # Massive matrix scale hardware stress tests
│   └── network.py         # Multi-Layer Perceptron (MLP) & Dense Layer implementations
│
├── Cargo.toml             # Rust crate configuration & dependencies
└── README.md              # Project documentation


🚀 Getting Started & Usage
Prerequisites
Python 3.8+ (with virtual environment .venv)

Rust Toolchain (Cargo)

Compatible GPU with compute capabilities

Installation & Execution

1. Activate your virtual environment:

source .venv/bin/activate  # On Linux/macOS
# Or on Windows PowerShell:
(.venv) .\.venv\Scripts\Activate.ps1

2. Run the performance benchmarks:

python src/benchmark.py

3. Run the hardware stress test:

python src/stress_test.py

4. Train the Neural Network (MLP):

python src/network.py




📄 License
This project is licensed under the PolyForm Noncommercial License 1.0.0. See the LICENSE file for details. Commercial use, modification, or distribution for commercial purposes is strictly prohibited without explicit written permission.

Built with precision, raw systems engineering, and GPU acceleration. 🦀🔥