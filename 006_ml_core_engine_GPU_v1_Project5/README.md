# ml_core_engine 🚀⚡

A high-performance, production-grade GPU-accelerated machine learning and linear algebra core engine built in **Rust** and exposed to **Python** via **PyO3**. 

Designed for raw speed and efficiency, this engine bypasses traditional CPU bottlenecks by leveraging **wgpu** and native **WGSL compute shaders** with a persistent singleton architecture optimized for modern hardware (NVIDIA RTX series).

---

## 🌟 Key Architectural Highlights

* **Persistent GPU Context (`OnceLock`)**: Eliminates the heavy overhead of re-initializing the GPU device, adapter, and queue on every matrix operation, keeping the hardware in a constant state of readiness.
* **GPU-Accelerated Core Operations**: Native GPU compute pipelines for Matrix Multiplication (`gpu_matmul`), forward activations (`gpu_relu`), and backward gradient passes (`gpu_relu_backward`).
* **Zero-Redundancy Design**: Streamlined codebase built for high maintainability, precision, and low-latency execution.
* **Seamless Python Integration**: High-performance bindings via `PyO3` allowing effortless integration into Python machine learning workflows.

---

## 🛠️ Tech Stack

* **Core Language**: Rust (Memory-safe, high-concurrency)
* **Compute API**: `wgpu` & custom WGSL Shaders
* **Python Bindings**: `PyO3` & `Maturin`
* **Data Management**: `bytemuck` for safe zero-copy casting

---

## 📁 Project Structure

```text
006_ml_core_engine_GPU_v1_Project5/
├── src/
│   ├── lib.rs          # Rust core engine, GPU context singleton, and PyO3 bindings
│   ├── network.py      # Neural network architecture (DenseLayer, SimpleMLP, Xavier init)
│   └── benchmark.py    # Performance benchmarking suite for GPU/CPU comparison
├── Cargo.toml          # Rust dependencies and crate configuration
├── pyproject.toml      # Maturin build system configuration
└── README.md           # Project documentation


⚙️ Installation & Development
Make sure you have Rust, Python, and Maturin installed on your system.

1. Clone the repository:

git clone https://github.com/GoldenFist06/ml-systems-playground.git
cd ml-systems-playground

2. Create and activate a Python virtual environment:

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

3. Build and install the engine in editable mode using Maturin:

maturin develop

___

🚀 Usage & Execution
1. Run Performance Benchmark
Test the raw GPU matrix multiplication and activation throughput:

python src/benchmark.py

2. Train a Neural Network (XOR / Custom MLP)
Run the end-to-end training pipeline utilizing GPU-accelerated forward and backward passes:

python src/network.py

Operation,Hardware,Execution Time
Matrix Multiplication (500×500),CPU Baseline,~2.204005s
Matrix Multiplication (500×500),GPU (wgpu Compute),~0.654s
"1,000 ReLU Iterations",GPU (Persistent Context),~3.219s






> **⚠️ Hardware & Benchmark Note:** 
> Performance metrics depend heavily on the underlying hardware. Testing was conducted on an **Intel Core i7 (13th Gen) CPU** (baseline from `005_ml_core_engine_Project4`) versus an **NVIDIA RTX 4050 GPU** (current GPU-accelerated engine), highlighting the massive throughput gains achieved through compute shaders and persistent context caching.




📄 License
This project is licensed under the PolyForm Noncommercial License 1.0.0.
See the LICENSE file for details. Commercial use, modification, or distribution for commercial purposes is strictly prohibited without explicit written permission.
