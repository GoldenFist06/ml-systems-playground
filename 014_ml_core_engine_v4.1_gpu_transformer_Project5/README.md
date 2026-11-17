# 🚀 ML Core Engine v4.1: High-Performance GPU-Accelerated Transformer Framework

An ultra-optimized, bare-metal machine learning and transformer inference core engine built from scratch using **Rust** for blazing-fast hardware execution and **Python** for high-level orchestration, bound via **PyO3/FFI**.

Designed to bypass the traditional overheads of deep learning interpreters, this engine achieves extreme-throughput matrix computations, native transformer primitives, thread-safe concurrency, and memory-efficient kernel operations entirely resident within the VRAM architecture.

## ⚠️ Note on Project Lineage

> This project is the heavily optimized, patched, and performance-upgraded evolution to **v4.1**, featuring enhanced mathematical precision audits and error-variance minimization, true asynchronous execution pipelines (eliminating synchronous `device.poll()` CPU stalls), Shared Memory 2D Block Tiling GEMM with 17-stride padding to eliminate bank conflicts, Online Softmax Fused Attention, Residual RMSNorm fusion, bulk `KvCache` updates, and a comprehensive dual-profile hardware benchmarking suite.

## 🏗️ Architectural Overview

Modern machine learning runtimes often suffer from latency overheads caused by frequent data transfers between host (CPU) and device (GPU) memory, synchronous driver stalls, and separate kernel execution passes for operations and activations.

This project tackles these bottlenecks through core innovations:

* **True Asynchronous Execution & Persistent VRAM Compute Engine:** Keeps the compute context alive within a thread-safe singleton (`OnceLock`), eliminates redundant `poll()` calls, and uses the Rust `Drop` trait to automatically reclaim VRAM buffers without host-device memory leaks.
* **Shared Memory 2D Block Tiling & Vectorized Layout:** Utilizes `vec4<f32>` vectorized memory loads, workgroup shared memory tiles with 17-stride padding to completely bypass bank conflicts, and optimized 2D workgroup dispatch grids to fully saturate GPU ALUs.
* **Transformer-Ready Native Kernels & Bulk KV Cache:** Implements foundational transformer blocks directly on GPU hardware with single-encoder bulk `KvCache` updates, `Residual + RMSNorm` fusion, and Online Softmax Fused Attention to accelerate autoregressive decoding and context prefill.
* **Ultra-Low Engine Overhead:** Achieves an exceptional cumulative scheduling overhead ratio of **4.92%** across core kernels, ensuring **95.08%** of engine lifecycle time is dedicated to pure GPU compute.

## 🛠️ Key Architectural Enhancements & Solved Bottlenecks

### 1. Mathematical Precision Audits & Error-Variance Minimization (`v4.1`)

* **The Problem:** High-performance GPU kernels often accumulate floating-point rounding errors or precision drift when executing deep reduction and matrix multiplication pipelines under massive parallelism.
* **The Solution:** Implemented rigorous mathematical audit samplers and high-precision verification tests (`engine_audit.py`) ensuring worst-case relative errors stay bounded within strict tolerances (e.g., matching NumPy GEMM outputs down to $5.24 \times 10^{-7}$ relative error).
* **The Impact:** Guarantees absolute numerical stability and deterministic floating-point precision (`FP32`) across all transformer primitives, activation layers, and multi-layer perceptron training backpropagation steps.

### 2. Asynchronous Command Batching & Shared Memory Padding (`v4.1`)

* **The Problem:** 16-stride memory access patterns in shared memory tiles can trigger severe bank conflicts, and blocking CPU synchronizations bottleneck throughput.
* **The Solution:** Upgraded `gpu_matmul` to use 17-stride shared memory padding to eliminate bank conflicts completely, alongside fully asynchronous command dispatching.
* **The Impact:** Smashed performance barriers, achieving sustained throughputs exceeding **2.3 TFLOPS** on standard benchmarks and scaling up to **~2.0 - 2.3+ TFLOPS** of pure hardware-verified VRAM compute capacity.

### 3. Full Transformer Primitives & Kernel Fusion Suite (`v4.1`)

* Extended the core engine with dedicated, fused GPU shaders for large language models:
  * `gpu_rms_norm` & `gpu_residual_rms_norm`: Tree-parallel reduction Root Mean Square Normalization with fused residual stream addition.
  * `gpu_swiglu`: Swish-Gated Linear Unit hybrid activation function.
  * `gpu_softmax`: Numerically stable parallel softmax layer distributed across workgroups with verified row-sum integrity ($\sum = 1.0000$).
  * `gpu_rope`: Rotary Position Embedding for spatial token transformations.
  * `gpu_fused_attention`: Single-pass Scaled Dot-Product Attention with Online Softmax backed by bulk-updated `KvCache` layers.

### 4. Zero-Overhead Python-Rust Bridge & Bulletproof Concurrency

* Leveraging **PyO3**, tensor matrices are managed directly in Rust's raw memory allocations while exposing an intuitive Python API (`PyMatrix`, `GpuMatrix`, and `KvCache`), verified via multi-threaded concurrency suites (`hardware_verification_suite.py`) supporting 32 simultaneous background threads with zero host RSS memory drift.

## 📊 Performance Benchmarks & Stress Testing

Comprehensive hardware stress tests conducted on consumer-grade hardware (**ASUS TUF F15 — NVIDIA GeForce RTX 4050 Laptop GPU**) demonstrate the engine's extreme saturation capabilities in **v4.1**:

| Benchmark / Operation | Payload Size / Configuration | Throughput / Bandwidth | Native Latency |
| :--- | :--- | :--- | :--- |
| **Pure VRAM GEMM Peak (8192 Scale)** | $8192 \times 8192$ Matrix (Persistent Buffers) | **2,025.45 GFLOPS (~2.03 TFLOPS)** | `0.54285 s` / loop |
| **Comprehensive Engine Audit Benchmark** | $4096 \times 4096$ Matrix | **2,326.93 GFLOPS (~2.33 TFLOPS)** | Verified Pass |
| **Multi-User Generation Throughput** | 24 Layers \| Hidden=4096 \| Batch=32 (4.69 GB) | **366.18 Total TPS** | `87.39 ms` / step |
| **Context Prefill Ingestion Speed** | 1-Layer Target (Extrapolated to 24 Layers) \| 2,048 Tokens | **11,539.73 Prefill TPS** | `7.39 ms` / layer step |
| **KV-Cache Update Bandwidth** | $7000 \times 128$ Vector Slice | **109.96 GB/s** | `0.0652 ms` |
| **Residual RMSNorm / SwiGLU / Add / Sub** | $7000 \times 7000$ (~186.92 MB) | **40 - 42 GB/s** | `~14.0 - 14.6 ms` |
| **Collective Engine Scheduling Overhead** | Core Kernels Profiling | **4.92%** (95.08% Pure Compute) | `4.66 ms` cumulative |

> **Key Takeaway from Stress Tests:** The v4.1 engine achieves absolute bare-metal saturation with an average 95.08% pure GPU compute utilization, running multi-user transformer workloads at **366.18 TPS** and context prefill ingestion at over **11,500 TPS** while maintaining rigorous mathematical precision.

## 📂 Project Structure

```text
013_ml_core_engine_v4.0_gpu_transformer_Project5/
│
├── src/
│   ├── lib.rs                                            # Rust core engine, WGSL tiled/fused shaders, KvCache, FFI bindings
│   ├── network.py                                        # Multi-Layer Perceptron (MLP) & Full Transformer Block implementations
│   ├── engine_audit.py                                   # Comprehensive black-box 10/10 NumPy correctness & audit suite
│   ├── hardware_verification_suite.py                    # Rigorous 32-thread concurrency & memory churn integrity suite
│   ├── test_e2e_comprehensive.py                         # End-to-end integration smoke test for all v4.0 primitives
│   ├── test_core_efficiency.py                           # Non-linear kernel bandwidth & mathematical audit sampler
│   ├── benchmark_all_kernels_perf.py                     # Sustained 100-iteration performance profiler across all 13 kernels
│   ├── benchmark_comprehensive_overhead.py               # Dual-profile CPU dispatch overhead vs GPU compute ratio profiler
│   ├── test_profile_raw_compute_saturation_flops.py      # Peak instantaneous raw compute saturation profiler (up to 11.4 TFLOPS)
│   ├── test_continuous_workload_bandwidth_flops.py       # Sustained thermal & memory-bus workload stress test
│   ├── stress_test.py                                    # Multi-dimension volumetric GEMM stress test
│   ├── test_ai_runtime_simulation.py                     # Sovereign stable transformer character runtime simulation
│   ├── test_engine.py                                    # Long-context (4000x4096) attention & bus transfer benchmark
│   ├── test_flash_softmax.py                             # Isolated Fused Attention + Online Softmax latency profiler
│   ├── test_generation_tps.py                            # Single-user (Batch=1) 24-layer 1.25B model latency benchmark
│   ├── benchmark_multi_user_throughput.py                # Multi-user (Batch=32) 24-layer 1.25B model throughput benchmark
│   ├── benchmark_prefill_context.py                      # Parallel context prefill ingestion benchmark (2048 tokens)
│   └── verify_mathematical_integrity.py                  # mathematical precision verification
│
├── Cargo.toml                                            # Rust crate configuration & release hardening profile
├── pyproject.toml                                        # Maturin python build configuration
└── README.md                                             # Project documentation
```


## 🚀 Getting Started & Usage

### Prerequisites

* Python 3.10+ (with virtual environment `.venv`)
* Rust Toolchain (`Cargo` and `Maturin`)
* Compatible GPU with Vulkan/DX12/Metal Compute capabilities
* Requires `numpy` and `psutil` (for RSS memory differential tracking)

### Installation & Compilation

1. **Activate your virtual environment (Windows PowerShell):**

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies via pip:**

   ```bash
   pip install numpy psutil maturin
   ```

3. **Install the core:**

   ```bash
   python -m pip install --force-reinstall "src/ml_core_engine-0.1.0-cp312-cp312-win_amd64.whl"
   ```


### Running Verification & Benchmark Suites

4. **Run the end-to-end comprehensive smoke test:**

   ```bash
   python src/test_e2e_comprehensive.py
   ```

5. **Run the 10/10 NumPy mathematical audit suite:**

   ```bash
   python src/engine_audit.py
   ```

6. **Run the 32-thread hardware integrity and concurrency suite:**

   ```bash
   python src/hardware_verification_suite.py
   ```

7. **Run the comprehensive kernel efficiency & bandwidth audit:**

   ```bash
   python src/test_core_efficiency.py
   ```

8. **Run the peak raw compute saturation FLOPS benchmark:**

   ```bash
   python src/test_profile_raw_compute_saturation_flops.py
   ```

9. **Run the sustained thermal & memory-bus workload FLOPS benchmark:**

   ```bash
   python src/test_continuous_workload_bandwidth_flops.py
   ```

10. **Run the single-user (Batch=1) 24-layer Transformer latency benchmark:**

    ```bash
    python src/test_generation_tps.py
    ```

11. **Run the multi-user (Batch=32) 24-layer Transformer throughput benchmark:**

    ```bash
    python src/benchmark_multi_user_throughput.py
    ```

12. **Run the parallel context prefill ingestion benchmark:**

    ```bash
    python src/benchmark_prefill_context.py
    ```

13. **Run the scheduling overhead profiler:**

    ```bash
    python src/benchmark_comprehensive_overhead.py
    ```

14. **Run the neural network (MLP) & Transformer block pipeline:**

    ```bash
    python src/network.py
    ```

15. **Run the ai chat simulation:** 

    ```bash
    python src/test_ai_runtime_simulation.py
    ```

16. **Run the  mathematical precision audit:** 

    ```bash
    python src/verify_mathematical_integrity.py
    ```

## 📄 License

This project is licensed under the PolyForm Noncommercial License 1.0.0. See the LICENSE file for details.
Commercial use, modification, or distribution for commercial purposes is strictly prohibited without explicit written permission.

🚀 GPU Device Active and Initialized via Persistent VRAM Compute Engine (v4.1 Active)!

⭐ Enjoying ml_core_engine? Support the project on GitHub: https://github.com/GoldenFist06/ml-systems-playground

Built with precision, raw systems engineering, and GPU acceleration. 🦀🔥
