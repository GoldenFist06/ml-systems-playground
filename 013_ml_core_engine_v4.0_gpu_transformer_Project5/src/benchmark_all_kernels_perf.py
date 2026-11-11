import time
import ml_core_engine as ml

def run_kernel_benchmark(name, low_dim_func, heavy_func, dim, is_compute_bound=True):
    print(f"• Profiling {name:<30} | Capacity Matrix: {dim}x{dim}...")
    
    
    for _ in range(5):
        warm_res = low_dim_func()
        if isinstance(warm_res, ml.GpuMatrix):
            _ = warm_res.buffer_id()
    
    
    num_loops = 100
    start_time = time.perf_counter()
    
    for _ in range(num_loops):
        result_gpu = heavy_func()
        
    
    if isinstance(result_gpu, ml.GpuMatrix):
        _ = result_gpu.to_cpu()
        
    total_elapsed = time.perf_counter() - start_time
    elapsed_per_loop = total_elapsed / num_loops
    elapsed_ms = elapsed_per_loop * 1000
    
    
    if is_compute_bound and "MatMul" in name:
        total_flops = 2.0 * (dim ** 3)
        throughput = (total_flops / (elapsed_per_loop * 1e9))
        metric_str = f"🚀 {throughput:.2f} GFLOPS"
    else:
        metric_str = f"⏱️ {elapsed_ms:.4f} ms/loop"
        
    return elapsed_ms, metric_str

def main():
    print("=" * 85)
    print("💎 ML_CORE_ENGINE - COMPREHENSIVE KERNEL HARDWARE PERFORMANCE BENCHMARK")
    print("=" * 85)
    
    device_status = ml.check_gpu_device()
    print(f"📌 Hardware Device: {device_status}")
    
    
    dim = 7000
    d_k = 128
    
    print(f"\nAllocating Persistent Matrix Buffers in VRAM context ({dim}x{dim})...")
    
    a_low = ml.PyMatrix(4, 4, 1.0).to_gpu()
    b_low = ml.PyMatrix(4, 4, 1.0).to_gpu()
    kv_low = ml.KvCache(10, 4)
    
    
    a_heavy = ml.PyMatrix(dim, dim, 1.0001).to_gpu()
    b_heavy = ml.PyMatrix(dim, dim, 0.9999).to_gpu()
    
    q_heavy = ml.PyMatrix(dim, d_k, 0.125).to_gpu()
    kv_heavy = ml.KvCache(512, d_k)
    
    print("\nInitiating sequential core profiling loop (100 iterations per kernel)...")
    print("-" * 85)
    
    
    kernels_to_benchmark = [
        ("MatMul (GEMM Core)", lambda: a_low.gpu_matmul(b_low, False), lambda: a_heavy.gpu_matmul(b_heavy, False), dim, True),
        ("Scale (Element-wise)", lambda: a_low.gpu_scale(2.0), lambda: a_heavy.gpu_scale(2.0), dim, False),
        ("Transpose (Memory Shuffle)", lambda: a_low.gpu_transpose(), lambda: a_heavy.gpu_transpose(), dim, False),
        ("Softmax (Shared Memory)", lambda: a_low.gpu_softmax(), lambda: a_heavy.gpu_softmax(), dim, False),
        ("RMSNorm (Vectorized)", lambda: a_low.gpu_rms_norm(1e-5), lambda: a_heavy.gpu_rms_norm(1e-5), dim, False),
        ("Residual RMSNorm (Fused)", lambda: a_low.gpu_residual_rms_norm(b_low, 1e-5), lambda: a_heavy.gpu_residual_rms_norm(b_heavy, 1e-5), dim, False),
        ("SwiGLU (Activation Layer)", lambda: a_low.gpu_swiglu(b_low), lambda: a_heavy.gpu_swiglu(b_heavy), dim, False),
        ("RoPE (Rotary Embeddings)", lambda: a_low.gpu_rope(), lambda: a_heavy.gpu_rope(), dim, False),
        ("Tensor Add", lambda: a_low.gpu_add(b_low), lambda: a_heavy.gpu_add(b_heavy), dim, False),
        ("Tensor Sub", lambda: a_low.gpu_sub(b_low), lambda: a_heavy.gpu_sub(b_heavy), dim, False),
        ("ReLU Backward (Gradient)", lambda: a_low.gpu_relu_backward(b_low), lambda: a_heavy.gpu_relu_backward(b_heavy), dim, False),
        ("KV-Cache Update", lambda: kv_low.update(0, a_low, b_low), lambda: kv_heavy.update(0, q_heavy, q_heavy), dim, False),
        ("Fused Attention (Decoder)", lambda: a_low.gpu_fused_attention(kv_low, 1), lambda: q_heavy.gpu_fused_attention(kv_heavy, 1), dim, False)
    ]
    
    results = {}
    for name, low_func, heavy_func, target_dim, is_compute in kernels_to_benchmark:
        _, metric_output = run_kernel_benchmark(name, low_func, heavy_func, target_dim, is_compute)
        results[name] = metric_output
        time.sleep(1) 
        
    print("\n" + "=" * 85)
    print("📊 PURE HARDWARE RUNTIME PERFORMANCE REPORT")
    print("=" * 85)
    print(f"{'Execution Pipeline Function':<40} | {'Pure Hardware Performance / Latency':<35}")
    print("-" * 85)
    
    for name, performance in results.items():
        print(f"• {name:<38} | {performance}")
        
    print("=" * 85)
    print("🟢 STATUS: Absolute Bare-Metal Performance Captured Successfully with Zero Blurs.")
    print("=" * 85)

if __name__ == "__main__":
    main()
