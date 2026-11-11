# 🚀 ML Core Engine v4.0: High-Performance GPU-Accelerated Transformer Framework

An ultra-optimized, bare-metal machine learning and transformer inference core engine built from scratch using **Rust** for blazing-fast hardware execution and **Python** for high-level orchestration, bound via **PyO3/FFI**.

Designed to bypass the traditional overheads of deep learning interpreters, this engine achieves extreme-throughput matrix computations, native transformer primitives, thread-safe concurrency, and memory-efficient kernel operations entirely resident within the VRAM architecture.

## ⚠️ Note on Project Lineage

> This project is the heavily optimized, patched, and performance-upgraded evolution to **v4.0**, featuring true asynchronous execution pipelines (eliminating synchronous `device.poll()` CPU stalls), Shared Memory 2D Block Tiling GEMM (`16x16` tiles), Online Softmax Fused Attention, Residual RMSNorm fusion, bulk `KvCache` updates, and a comprehensive dual-profile hardware benchmarking suite (distinguishing between Raw Compute Saturation and Sustained Thermal Workloads).

## 🏗️ Architectural Overview

Modern machine learning runtimes often suffer from latency overheads caused by frequent data transfers between host (CPU) and device (GPU) memory, synchronous driver stalls, and separate kernel execution passes for operations and activations.

This project tackles these bottlenecks through core innovations:

* **True Asynchronous Execution & Persistent VRAM Compute Engine:** Keeps the compute context alive within a thread-safe singleton (`OnceLock`), eliminates redundant `poll()` calls, and uses the Rust `Drop` trait to automatically reclaim VRAM buffers without host-device memory leaks.
* **Shared Memory 2D Block Tiling & Vectorized Layout:** Utilizes `vec4<f32>` vectorized memory loads, workgroup shared memory tiles (`tile_a`, `tile_b`), and 2D workgroup dispatch grids (`@workgroup_size(16, 16)`) to bypass global VRAM bandwidth bottlenecks and fully saturate GPU ALUs.
* **Transformer-Ready Native Kernels & Bulk KV Cache:** Implements foundational transformer blocks directly on GPU hardware with single-encoder bulk `KvCache` updates, `Residual + RMSNorm` fusion, and Online Softmax Fused Attention to accelerate autoregressive decoding and context prefill.
* **Ultra-Low Engine Overhead:** Achieves an exceptional cumulative scheduling overhead ratio of **2.12%** across 13 distinct kernels, ensuring **97.88%** of engine lifecycle time is dedicated to pure GPU compute.

## 🛠️ Key Architectural Enhancements & Solved Bottlenecks

### 1. Asynchronous Command Batching & 2D Shared Memory Tiling (`v4.0`)

* **The Problem:** Synchronous GPU polling between dispatches and repeated global VRAM reads during matrix multiplication severely bottlenecked compute throughput in v3.9 (~2.6 TFLOPS ceiling).
* **The Solution:** Removed blocking `poll()` barriers during async execution, implemented `16x16` shared memory workgroup tiling in `gpu_matmul`, and migrated all core operations (`RMSNorm`, `ResidualRMSNorm`, `SwiGLU`, `Sub`, `ReLUBackward`, `Scale`, `Transpose`) to vectorized `vec4<f32>` pipelines.
* **The Impact:** More than doubled sustained GEMM throughput to **~4.97 - 5.77 TFLOPS** and unlocked **11.40 TFLOPS** peak raw compute utilization.

### 2. Full Transformer Primitives & Kernel Fusion Suite (`v4.0`)

* Extended the core engine with dedicated, fused GPU shaders for large language models:
  * `gpu_rms_norm` & `gpu_residual_rms_norm`: Tree-parallel reduction Root Mean Square Normalization with fused residual stream addition.
  * `gpu_swiglu`: Swish-Gated Linear Unit hybrid activation function.
  * `gpu_softmax`: Numerically stable parallel softmax layer distributed across workgroups.
  * `gpu_rope`: Rotary Position Embedding for spatial token transformations.
  * `gpu_fused_attention`: Single-pass Scaled Dot-Product Attention with Online Softmax backed by bulk-updated `KvCache` layers.

### 3. Zero-Overhead Python-Rust Bridge

* Leveraging **PyO3**, tensor matrices are managed directly in Rust's raw, safe memory allocations while exposing an intuitive, native Python API (`PyMatrix`, `GpuMatrix`, and `KvCache`).

### 4. Bulletproof Concurrency & Memory Stability

* Verified through rigorous multi-threaded concurrency suites (`hardware_verification_suite.py`), executing 32 simultaneous background threads hammering the GPU concurrently with **+0.02 MB host RSS memory differential** and robust isolated exception handling.

## 📊 Performance Benchmarks & Stress Testing

Comprehensive hardware stress tests conducted on consumer-grade laptop hardware (**ASUS TUF F15 — NVIDIA GeForce RTX 4050 Laptop GPU, 96-bit Memory Bus**) demonstrate the engine's saturation capabilities in **v4.0**, distinguishing clearly between raw compute saturation and sustained thermal/bandwidth-bound workloads:

| Benchmark / Operation | Payload Size / Configuration | Throughput / Bandwidth | Native Latency |
| :--- | :--- | :--- | :--- |
| **Raw Compute Saturation (Peak Burst)** | $7000 \times 7000$ Matrix (Persistent Buffers) | **11,401.06 GFLOPS (~11.40 TFLOPS)** | `60.17 ms` |
| **Balanced GEMM Peak (50-Loop Profile)** | $7000 \times 7000$ Matrix (~186.92 MB) | **5,765.62 GFLOPS (~5.77 TFLOPS)** | `118.98 ms` |
| **Sustained GEMM Compute (100-Loop TGP)** | $6000 \times 6000$ / $7000 \times 7000$ Matrix | **4,974.57 GFLOPS (~4.97 TFLOPS)** | `86.84 ms` |
| **Sustained Volumetric GEMM** | $8192 \times 8192$ Matrix (~256.00 MB) | **4,629.53 GFLOPS (~4.63 TFLOPS)** | `237.50 ms` |
| **Multi-User Generation (Throughput)** | 24 Layers \| Hidden=4096 \| Batch=32 (4.69 GB) | **982.75 Total TPS** | `32.56 ms` / step |
| **Single-User Generation (Latency)** | 24 Layers \| Hidden=4096 \| Batch=1 (4.69 GB) | **28.02 Pure TPS** | `35.68 ms` / token |
| **Context Prefill Ingestion** | 24 Layers \| 2,048 Parallel Tokens | **499.22 Prefill TPS** | `170.93 ms` / layer |
| **Fused Attention + Online Softmax** | Single-Head Isolated Cache ($Context=512, d_k=128$) | **Memory-Bound Fused Pass** | `0.0547 ms` (`54.7 µs`) |
| **Long-Context Attention Engine** | $4000 \times 4096$ Context Window | **177.12 GFLOPS** | `0.3700 ms` |
| **KV-Cache Bulk Update** | $7000 \times 128$ Vector Slice | **127.49 GB/s** | `0.0562 ms` |
| **Residual RMSNorm (Fused)** | $7000 \times 7000$ (~186.92 MB) | **75.55 GB/s** | `7.78 ms` |
| **ReLU Backward (Gradient)** | $7000 \times 7000$ (~186.92 MB) | **73.61 GB/s** | `7.99 ms` |
| **Tensor Sub / Tensor Add** | $7000 \times 7000$ (~186.92 MB) | **73.45 GB/s** | `8.01 ms` |
| **SwiGLU Activation** | $7000 \times 7000$ (~186.92 MB) | **73.32 GB/s** | `8.02 ms` |
| **Softmax (Shared Memory)** | $7000 \times 7000$ (~186.92 MB) | **54.22 GB/s** | `7.23 ms` |
| **RoPE (Rotary Embeddings)** | $7000 \times 7000$ (~186.92 MB) | **54.04 GB/s** | `7.25 ms` |
| **Collective Engine Scheduling Overhead** | 13 Kernels $\times$ 100 Iterations | **2.12%** (97.88% Compute) | `1.48 ms` cumulative |

> **Key Takeaway from Stress Tests:** Removing blocking CPU synchronization and introducing `16x16` shared memory tiling unlocked **5.77 TFLOPS** in balanced execution and **11.40 TFLOPS** in raw compute bursts. On a full 1.25B-parameter (4.69 GB FP32) 24-layer Transformer stack, the engine sustains **28.02 TPS** for single-user decoding and scales to **982.75 TPS** across 32 concurrent users.

## ⚠️ Hardware Performance Disclaimer

> **Tested Environment:** The performance benchmarks and stress tests documented in this project (reaching **5,765.62 GFLOPS** balanced and **11,401.06 GFLOPS** peak raw compute) were executed and verified on an **ASUS TUF F15 (NVIDIA GeForce RTX 4050 Laptop GPU, 96-bit Memory Bus)** via the Vulkan/wgpu backend.
>
> * **Performance Scaling Note:** Hardware acceleration, throughput (GFLOPS/TPS), thermal throttling boundaries, and memory bandwidth saturation heavily depend on your specific GPU architecture, VRAM bus width, core count, and cooling specifications.

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
│   ├── test_engine.py                                    # Long-context (4000x4096) attention & bus transfer benchmark
│   ├── test_flash_softmax.py                             # Isolated Fused Attention + Online Softmax latency profiler
│   ├── test_generation_tps.py                            # Single-user (Batch=1) 24-layer 1.25B model latency benchmark
│   ├── benchmark_multi_user_throughput.py                # Multi-user (Batch=32) 24-layer 1.25B model throughput benchmark
│   └── benchmark_prefill_context.py                      # Parallel context prefill ingestion benchmark (2048 tokens)
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


## 📄 License

This project is licensed under the PolyForm Noncommercial License 1.0.0. See the LICENSE file for details.
Commercial use, modification, or distribution for commercial purposes is strictly prohibited without explicit written permission.

🚀 GPU Device Active and Initialized via Persistent VRAM Compute Engine (v4.0 Active)!

⭐ Enjoying ml_core_engine? Support the project on GitHub:  https://github.com/GoldenFist06/ml-systems-playground

Built with precision, raw systems engineering, and GPU acceleration. 🦀🔥
