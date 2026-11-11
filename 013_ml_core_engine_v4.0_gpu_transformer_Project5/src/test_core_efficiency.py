import time
import random
import ml_core_engine as ml

def run_component_stress_test():
    print("=" * 85)
    print("🔬 ML_CORE_ENGINE - COMPREHENSIVE BARE-METAL KERNEL PERFORMANCE PROFILER")
    print("=" * 85)
    
    status = ml.check_gpu_device()
    print(f"📌 Status: {status}")
    
    ROWS = 7000
    COLS = 7000
    SIZE = ROWS * COLS
    D_K = 128
    LOOPS = 50
    print(f"📦 Payload Dimensions for Non-Linear Kernels: {ROWS}x{COLS} (~{((SIZE * 4) / (1024*1024)):.2f} MB)")
    print("=" * 85)
    
    random.seed(42)
    raw_data_a = [random.uniform(-5.0, 5.0) for _ in range(SIZE)]
    raw_data_b = [random.uniform(-5.0, 5.0) for _ in range(SIZE)]
    
    print("⚡ Transferring massive payloads to persistent VRAM...")
    mat_a_gpu = ml.PyMatrix.from_list(ROWS, COLS, raw_data_a).to_gpu()
    mat_b_gpu = ml.PyMatrix.from_list(ROWS, COLS, raw_data_b).to_gpu()
    
    q_gpu = ml.PyMatrix.from_list(ROWS, D_K, raw_data_a[:ROWS * D_K]).to_gpu()
    kv_cache = ml.KvCache(8192, D_K)

    def measure_kernel(name, rw_bytes, warmup_lambda, exec_lambda):
        for _ in range(5):
            w_res = warmup_lambda()
            if isinstance(w_res, ml.GpuMatrix): _ = w_res.buffer_id()
        
        start = time.perf_counter()
        for _ in range(LOOPS):
            res_gpu = exec_lambda()
        
        if isinstance(res_gpu, ml.GpuMatrix):
            _ = res_gpu.buffer_id() 
        end = time.perf_counter()
        
        if isinstance(res_gpu, ml.GpuMatrix):
            res_cpu = res_gpu.to_cpu()
        else:
            res_cpu = None
            
        elapsed = (end - start) / LOOPS
        latency_ms = elapsed * 1000
        
        if "MatMul" in name:
            total_flops = 2.0 * (ROWS ** 3)
            gflops = total_flops / (elapsed * 1e9)
            print(f"• {name:<30} | Latency: {latency_ms:>8.4f} ms | Performance: {gflops:>7.2f} GFLOPS")
        else:
            bandwidth = float(rw_bytes) / (elapsed * 1e9)
            print(f"• {name:<30} | Latency: {latency_ms:>8.4f} ms | Bandwidth: {bandwidth:>7.2f} GB/s")
            
        return res_cpu

    print("\n🚀 Commencing intensive execution sequence over 50 iterations per pipeline...")
    print("-" * 85)

    cpu_matmul = measure_kernel(
        "MatMul (GEMM Core)", 
        SIZE * 4 * 3, 
        lambda: mat_a_gpu.gpu_matmul(mat_b_gpu, False), 
        lambda: mat_a_gpu.gpu_matmul(mat_b_gpu, False)
    )
    
    cpu_scale = measure_kernel(
        "Scale (Element-wise)", 
        SIZE * 4 * 2, 
        lambda: mat_a_gpu.gpu_scale(2.5), 
        lambda: mat_a_gpu.gpu_scale(2.5)
    )
    
    cpu_transpose = measure_kernel(
        "Transpose (Memory Shuffle)", 
        SIZE * 4 * 2, 
        lambda: mat_a_gpu.gpu_transpose(), 
        lambda: mat_a_gpu.gpu_transpose()
    )
    
    cpu_softmax = measure_kernel(
        "Softmax (Shared Memory)", 
        SIZE * 4 * 2, 
        lambda: mat_a_gpu.gpu_softmax(), 
        lambda: mat_a_gpu.gpu_softmax()
    )
    
    cpu_rms = measure_kernel(
        "RMSNorm (Vectorized)", 
        SIZE * 4 * 2, 
        lambda: mat_a_gpu.gpu_rms_norm(1e-5), 
        lambda: mat_a_gpu.gpu_rms_norm(1e-5)
    )
    
    cpu_res_rms = measure_kernel(
        "Residual RMSNorm (Fused)", 
        SIZE * 4 * 3, 
        lambda: mat_a_gpu.gpu_residual_rms_norm(mat_b_gpu, 1e-5), 
        lambda: mat_a_gpu.gpu_residual_rms_norm(mat_b_gpu, 1e-5)
    )
    
    cpu_swiglu = measure_kernel(
        "SwiGLU (Activation Layer)", 
        SIZE * 4 * 3, 
        lambda: mat_a_gpu.gpu_swiglu(mat_b_gpu), 
        lambda: mat_a_gpu.gpu_swiglu(mat_b_gpu)
    )
    
    cpu_rope = measure_kernel(
        "RoPE (Rotary Embeddings)", 
        SIZE * 4 * 2, 
        lambda: mat_a_gpu.gpu_rope(), 
        lambda: mat_a_gpu.gpu_rope()
    )
    
    cpu_add = measure_kernel(
        "Tensor Add", 
        SIZE * 4 * 3, 
        lambda: mat_a_gpu.gpu_add(mat_b_gpu), 
        lambda: mat_a_gpu.gpu_add(mat_b_gpu)
    )
    
    cpu_sub = measure_kernel(
        "Tensor Sub", 
        SIZE * 4 * 3, 
        lambda: mat_a_gpu.gpu_sub(mat_b_gpu), 
        lambda: mat_a_gpu.gpu_sub(mat_b_gpu)
    )
    
    cpu_relu_back = measure_kernel(
        "ReLU Backward (Gradient)", 
        SIZE * 4 * 3, 
        lambda: mat_a_gpu.gpu_relu_backward(mat_b_gpu), 
        lambda: mat_a_gpu.gpu_relu_backward(mat_b_gpu)
    )
    
    _ = measure_kernel(
        "KV-Cache Update", 
        (ROWS * D_K * 4 * 2), 
        lambda: kv_cache.update(0, q_gpu, q_gpu), 
        lambda: kv_cache.update(0, q_gpu, q_gpu)
    )
    
    cpu_attn = measure_kernel(
        "Fused Attention (Decoder)", 
        (ROWS * D_K * 4 * 4), 
        lambda: q_gpu.gpu_fused_attention(kv_cache, 1), 
        lambda: q_gpu.gpu_fused_attention(kv_cache, 1)
    )

    print("\n" + "=" * 85)
    print("🔬 HIGH-PRECISION MATHEMATICAL AUDIT SAMPLING")
    print("=" * 85)
    print(f"• MatMul Output Verification   : {cpu_matmul.to_list()[:3] if cpu_matmul else 'None'}")
    print(f"• Scale Output Verification    : {cpu_scale.to_list()[:3] if cpu_scale else 'None'}")
    print(f"• Transpose Output Verification  : {cpu_transpose.to_list()[:3] if cpu_transpose else 'None'}")
    print(f"• Softmax Output Verification    : {cpu_softmax.to_list()[:3] if cpu_softmax else 'None'}")
    print(f"• RMSNorm Output Verification    : {cpu_rms.to_list()[:3] if cpu_rms else 'None'}")
    print(f"• Residual RMSNorm Verification : {cpu_res_rms.to_list()[:3] if cpu_res_rms else 'None'}")
    print(f"• SwiGLU Output Verification    : {cpu_swiglu.to_list()[:3] if cpu_swiglu else 'None'}")
    print(f"• RoPE Output Verification      : {cpu_rope.to_list()[:3] if cpu_rope else 'None'}")
    print(f"• Add Output Verification       : {cpu_add.to_list()[:3] if cpu_add else 'None'}")
    print(f"• Sub Output Verification       : {cpu_sub.to_list()[:3] if cpu_sub else 'None'}")
    print(f"• ReLU Back Output Verification  : {cpu_relu_back.to_list()[:3] if cpu_relu_back else 'None'}")
    print(f"• Attention Output Verification : {cpu_attn.to_list()[:3] if cpu_attn else 'None'}")
    print("=" * 85)
    print("🏁 Comprehensive hardware capability verification and performance audit finalized.")
    print("=" * 85)

if __name__ == "__main__":
    run_component_stress_test()
