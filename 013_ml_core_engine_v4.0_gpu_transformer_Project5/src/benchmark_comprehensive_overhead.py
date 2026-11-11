import time
import ml_core_engine

def analyze_comprehensive_overhead():
    print("=" * 85)
    print("🔬 ML_CORE_ENGINE - COMPREHENSIVE OVERHEAD PERCENTAGE PROFILER")
    print("=" * 85)
    
    status_msg = ml_core_engine.check_gpu_device()
    print(f" {status_msg}")

    overhead_dim = 4      
    heavy_dim = 4096      
    iterations = 100

    print(f"\n Preparing Tensors in VRAM context...")
    a_low = ml_core_engine.PyMatrix(overhead_dim, overhead_dim, 2.0).to_gpu()
    b_low = ml_core_engine.PyMatrix(overhead_dim, overhead_dim, 3.0).to_gpu()
    kv_low = ml_core_engine.KvCache(10, 4)

    a_heavy = ml_core_engine.PyMatrix(heavy_dim, heavy_dim, 0.5).to_gpu()
    b_heavy = ml_core_engine.PyMatrix(heavy_dim, heavy_dim, 0.2).to_gpu()
    
    d_k_heavy = 128
    q_heavy = ml_core_engine.PyMatrix(heavy_dim, d_k_heavy, 0.1).to_gpu()
    kv_heavy = ml_core_engine.KvCache(512, d_k_heavy)

    print(" Warming up internal execution pipelines...")
    for _ in range(5):
        _ = a_low.gpu_matmul(b_low, False)
        _ = a_low.gpu_scale(2.0)
        _ = a_low.gpu_transpose()
        _ = a_low.gpu_softmax()
        _ = a_low.gpu_rms_norm(1e-5)
        _ = a_low.gpu_residual_rms_norm(b_low, 1e-5)
        _ = a_low.gpu_swiglu(b_low)
        _ = a_low.gpu_rope()
        _ = a_low.gpu_add(b_low)
        _ = a_low.gpu_sub(b_low)
        _ = a_low.gpu_relu_backward(b_low)
        kv_low.update(0, a_low, b_low)
        _ = a_low.gpu_fused_attention(kv_low, 1)

    print(f" Running dual-profile timing loop over {iterations} iterations...")

    pure_overhead = {}
    total_compute_time = {}

    kernels_to_profile = [
        ("MatMul (GEMM Core)", lambda: a_low.gpu_matmul(b_low, False), lambda: a_heavy.gpu_matmul(b_heavy, False)),
        ("Scale (Element-wise)", lambda: a_low.gpu_scale(2.0), lambda: a_heavy.gpu_scale(2.0)),
        ("Transpose (Memory Shuffle)", lambda: a_low.gpu_transpose(), lambda: a_heavy.gpu_transpose()),
        ("Softmax (Shared Memory)", lambda: a_low.gpu_softmax(), lambda: a_heavy.gpu_softmax()),
        ("RMSNorm (Vectorized)", lambda: a_low.gpu_rms_norm(1e-5), lambda: a_heavy.gpu_rms_norm(1e-5)),
        ("Residual RMSNorm (Fused)", lambda: a_low.gpu_residual_rms_norm(b_low, 1e-5), lambda: a_heavy.gpu_residual_rms_norm(b_heavy, 1e-5)),
        ("SwiGLU (Activation Layer)", lambda: a_low.gpu_swiglu(b_low), lambda: a_heavy.gpu_swiglu(b_heavy)),
        ("RoPE (Rotary Embeddings)", lambda: a_low.gpu_rope(), lambda: a_heavy.gpu_rope()),
        ("Tensor Add", lambda: a_low.gpu_add(b_low), lambda: a_heavy.gpu_add(b_heavy)),
        ("Tensor Sub", lambda: a_low.gpu_sub(b_low), lambda: a_heavy.gpu_sub(b_heavy)),
        ("ReLU Backward (Gradient)", lambda: a_low.gpu_relu_backward(b_low), lambda: a_heavy.gpu_relu_backward(b_heavy)),
        ("KV-Cache Update", lambda: kv_low.update(0, a_low, b_low), lambda: kv_heavy.update(0, q_heavy, q_heavy)),
        ("Fused Attention (Decoder)", lambda: a_low.gpu_fused_attention(kv_low, 1), lambda: q_heavy.gpu_fused_attention(kv_heavy, 1))
    ]

    for name, low_func, heavy_func in kernels_to_profile:
        t0 = time.perf_counter()
        for _ in range(iterations): res_l = low_func()
        if isinstance(res_l, ml_core_engine.GpuMatrix): _ = res_l.buffer_id()
        pure_overhead[name] = ((time.perf_counter() - t0) * 1000) / iterations

        t1 = time.perf_counter()
        for _ in range(iterations): res_h = heavy_func()
        if isinstance(res_h, ml_core_engine.GpuMatrix): 
            _ = res_h.to_cpu()
        total_compute_time[name] = ((time.perf_counter() - t1) * 1000) / iterations

    print("\n" + "=" * 85)
    print("📊 CORRECTED ARCHITECTURAL OVERHEAD & UTILIZATION PROFILER REPORT")
    print("=" * 85)
    print(f"{'Kernel Function':<28} | {'Dispatch (ms)':<15} | {'Total (ms)':<12}")
    print("-" * 85)
    
    total_pure_dispatch_ms = 0.0
    total_engine_run_ms = 0.0

    for name in pure_overhead.keys():
        overhead_ms = pure_overhead[name]
        total_ms = total_compute_time[name]
        
        total_pure_dispatch_ms += overhead_ms
        total_engine_run_ms += total_ms
        
        print(f"• {name:<26} | {overhead_ms:.4f} ms    | {total_ms:.3f} ms")

    global_overhead_ratio = (total_pure_dispatch_ms / total_engine_run_ms) * 100.0
    
    print("=" * 85)
    print(f"• Total Cumulative Dispatch Latency    : ⚡ {total_pure_dispatch_ms:.4f} ms")
    print(f"• Total Engine Workload Execution Time  : 📊 {total_engine_run_ms:.3f} ms")
    print(f"• Collective Core Overhead Ratio (Avg)  : 🚀 {global_overhead_ratio:.2f}% of engine lifecycle")
    print(f"• Silicon Workload Saturation Capacity  : ⚡ {100.0 - global_overhead_ratio:.2f}% Pure GPU Compute")
    print("=" * 85)

    if global_overhead_ratio < 1.0:
        print("🟢 STATUS: Absolute Sovereign Efficiency. The CPU scheduling footprint is practically zero.")
    elif global_overhead_ratio < 5.0:
        print("🟡 STATUS: Highly Efficient Production Level. Micro-latency is well bounded.")
    else:
        print("🔴 STATUS: Bounded Bureaucracy. CPU dispatch is choking the compute pipeline.")
    print("=" * 85)

if __name__ == "__main__":
    analyze_comprehensive_overhead()
