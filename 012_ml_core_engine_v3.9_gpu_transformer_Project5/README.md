# 🚀 ML Core Engine v3.9: High-Performance GPU-Accelerated Transformer Framework

An ultra-optimized, GPU-optimized machine learning and transformer inference core engine built from scratch using **Rust** for blazing-fast hardware execution and **Python** for high-level orchestration, bound via **PyO3/FFI**.

Designed to bypass the traditional overheads of deep learning interpreters, this engine achieves extreme-throughput matrix computations, native transformer primitives, thread-safe concurrency, and memory-efficient kernel operations entirely resident within the VRAM architecture.

## ⚠️ Note on Project Lineage

> This project is the heavily optimized, patched, and performance-upgraded evolution to **v3.9**, featuring advanced architectural redesigns like full migration to `vec4<f32>` vectorized kernels across all operations, 2D Workgroup Dispatch grids, an Auto-Drop VRAM Garbage Collector, integrated `KvCache` architecture for autoregressive transformer decoding, and rigorous hardware integrity compliance suites.

## 🏗️ Architectural Overview

Modern machine learning runtimes often suffer from latency overheads caused by frequent data transfers between host (CPU) and device (GPU) memory, as well as separate kernel execution passes for operations and activations.

This project tackles these bottlenecks through core innovations:

* **Persistent VRAM Compute Engine & Auto-Drop Garbage Collector:** Keeps the compute context alive and persistent within a thread-safe singleton (`OnceLock`), while the Rust `Drop` trait automatically reclaims VRAM buffers without host-device memory leaks.

* **True Vectorized 2D Dispatch Layout:** Utilizes `vec4<f32>` vectorized memory loads and 2D workgroup dispatch grids (`@workgroup_size(16, 16)`) to bypass hardware limits ($65,535$ workgroups) and fully saturate GPU ALUs.

* **Transformer-Ready Native Kernels & KV Cache:** Implements foundational transformer blocks directly on GPU hardware with native `KvCache` integration and Fused Attention to accelerate autoregressive token generation.

* **Minimal Engine Overhead:** Achieves an ultra-low scheduling overhead ratio of **~0.14%**, ensuring 99.86% of GPU time is dedicated entirely to raw hardware computation.

## 🛠️ Key Architectural Enhancements & Solved Bottlenecks

### 1. Fully Vectorized `vec4<f32>` Kernels & Kernel Fusion

* **The Problem:** Non-vectorized scalar operations and intermediate VRAM read/write round-trips create severe memory bandwidth bottlenecks.

* **The Solution:** Migrated all core operations (`RMSNorm`, `SwiGLU`, `Sub`, `ReLUBackward`, `MatMul`) to vectorized `vec4<f32>` pipelines, paired with fused WGSL compute shaders (e.g., `gpu_matmul(..., use_relu=True)`) that apply activations on-the-fly inside GPU registers/shared memory.

* **The Impact:** Drastically slashes memory traffic and maximizes hardware saturation.

### 2. Full Transformer Primitives Suite (`v3.9`)

* Extended the core engine with dedicated, highly optimized GPU shaders for large language models:

  * `gpu_rms_norm`: Tree-parallel reduction Root Mean Square Normalization.

  * `gpu_swiglu`: Swish-Gated Linear Unit hybrid activation function.

  * `gpu_softmax`: Numerically stable parallel softmax layer distributed across workgroups.

  * `gpu_rope`: Rotary Position Embedding for spatial token transformations.

  * `gpu_fused_attention`: Scaled Dot-Product Attention backed by persistent `KvCache` layers.

### 3. Zero-Overhead Python-Rust Bridge

* Leveraging **PyO3**, tensor matrices are managed directly in Rust's raw, safe memory allocations while exposing an intuitive, native Python API (`PyMatrix`, `GpuMatrix`, and `KvCache`).

### 4. Bulletproof Concurrency & Memory Stability

* Verified through rigorous multi-threaded concurrency suites (`hardware_verification_suite.py`), executing 32 simultaneous background threads hammering the GPU concurrently with **+0.00 MB host RSS memory differential** and robust isolated exception handling.

## 📊 Performance Benchmarks & Stress Testing

Recent extreme hardware stress tests conducted on consumer-grade laptop hardware (NVIDIA GeForce RTX 4050 GPU) demonstrate the engine's true saturation capabilities in v3.9:

| Benchmark / Operation | Payload Size / Dimensions | Throughput / Latency |
| :--- | :--- | :--- |
| **Max GEMM Compute (Peak TFLOPS)** | $7000 \times 7000$ Matrix | **2,612.12 GFLOPS (~2.6 TFLOPS)** |
| **SwiGLU Activation** | $8192 \times 8192$ (~268 MB) | **41.89 GB/s** (Latency: `19.22 ms`) |
| **Element-wise Add** | $8192 \times 8192$ (~268 MB) | **37.58 GB/s** (Latency: `21.43 ms`) |
| **RMSNorm (Tree-Parallel)** | $8192 \times 8192$ (~268 MB) | **29.63 GB/s** (Latency: `18.12 ms`) |
| **RoPE (Rotary Embeddings)** | $8192 \times 8192$ (~268 MB) | **23.55 GB/s** (Latency: `22.79 ms`) |
| **Fused Attention + KV Cache** | $4000 \times 4000$ Context | **120.09 ms** (Autoregressive Decode Step) |
| **Engine Scheduling Overhead** | $4000 \times 4000$ MatMul | **0.14%** (Dispatch latency: `0.15 ms`) |

> **Key Takeaway from Stress Tests:** The GPU achieves complete hardware saturation at scale, peaking at an impressive **2,612.12 GFLOPS (~2.6 TFLOPS)** at the $\$7000 \times 7000\$$ scale. All transformer primitives operate with zero CPU overhead directly inside the VRAM space.

## ⚠️ Hardware Performance Disclaimer

> **Tested Environment:** The performance benchmarks and stress tests documented in this project (reaching peak hardware saturation of **2,612.12 GFLOPS**) were executed and verified on an **NVIDIA GeForce RTX 4050 Laptop GPU** via the Vulkan/wgpu backend.
>
> * **Performance Scaling Note:** Hardware acceleration, throughput (GFLOPS), and execution times heavily depend on your specific GPU architecture, VRAM bandwidth, core count, and system specifications.

## 📂 Project Structure

```
012_ml_core_engine_v3.9_gpu_transformer_Project5/
│
├── src/
│   ├── lib.rs                         # Rust core engine, WGSL vectorized shaders, KvCache, FFI bindings
│   ├── network.py                     # Multi-Layer Perceptron (MLP) & Transformer Block implementations
│   ├── benchmark.py                   # Standard and Fused kernel performance checks
│   ├── stress_test.py                 # Massive matrix scale hardware stress tests
│   ├── test_engine.py                 # Transformer attention benchmark & pipeline verification
│   ├── test_gflops.py                 # Max intensity GFLOPS & precision stress test
│   ├── test_transfer_overhead.py      # Host-device transfer overhead & FFI latency profiling
│   ├── test_flash_softmax.py          # Realistic attention stress benchmark with KV Cache
│   ├── test_core_efficiency.py        # Non-linear kernel throughput and bandwidth audit
│   ├── test_e2e_comprehensive.py      # End-to-end integration smoke test for all v3.9 primitives
│   ├── engine_audit.py                # Comprehensive advanced black-box 10/10 audit suite
│   ├── hardware_verification_suite.py # Rigorous 32-thread concurrency & memory churn integrity suite
│   ├── test_transfer_overhead.py      # Host-device transfer overhead & FFI latency profiling
│   └── test_remaining_kernels.py      # Verification suite for auxiliary & remaining kernels (sub, scale, transpose, etc.)
│
├── Cargo.toml                         # Rust crate configuration & dependencies
├── pyproject.toml                     # Maturin python build configuration
└── README.md                          # Project documentation
```

## 🚀 Getting Started & Usage

### Prerequisites

* Python 3.10+ (with virtual environment `.venv`)

* Rust Toolchain (`Cargo` and `Maturin`)

* Compatible GPU with Vulkan/Compute capabilities

* Requires NumPy and psutil (optional for RSS differential tracking)

### Installation & Execution

1. **Activate your virtual environment:**

   ```bash
   # On Windows PowerShell:
   (.venv) .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies via pip:**

   ```bash
   pip install numpy psutil
   ```

3. **Install the core:**

   ```bash
   python -m pip install --force-reinstall "src/ml_core_engine-0.1.0-cp312-cp312-win_amd64.whl"
   ```

4. **Run the end-to-end comprehensive smoke test:**

   ```bash
   python src/test_e2e_comprehensive.py
   ```

5. **Run the comprehensive audit & transformer verification suite:**

   ```bash
   python src/engine_audit.py
   ```

6. **Run the rigorous hardware integrity and concurrency suite:**

   ```bash
   python src/hardware_verification_suite.py
   ```

7. **Run the performance benchmarks:**

   ```bash
   python src/benchmark.py
   ```

8. **Run the massive hardware stress test:**

   ```bash
   python src/stress_test.py
   ```

9. **Run peak GFLOPS intensity test:**

   ```bash
   python src/test_gflops.py
   ```

10. **Run transfer overhead & latency profiling:**

    ```bash
    python src/test_transfer_overhead.py
    ```

11. **Run auxiliary and remaining kernels verification test:**

    ```bash
    python src/test_remaining_kernels.py
    ```

12. **Train and verify the Neural Network (MLP) & Transformer Block:**

    ```bash
    python src/network.py
    ```

## 📄 License

This project is licensed under the PolyForm Noncommercial License 1.0.0. See the LICENSE file for details.
Commercial use, modification, or distribution for commercial purposes is strictly prohibited without explicit written permission.

🚀 GPU Device Active and Initialized via Persistent VRAM Compute Engine (v3.9 Active)!

⭐ Enjoying ml_core_engine? Support the project on GitHub: https://github.com/GoldenFist06/ml-systems-playground

Built with precision, raw systems engineering, and GPU acceleration. 🦀🔥
