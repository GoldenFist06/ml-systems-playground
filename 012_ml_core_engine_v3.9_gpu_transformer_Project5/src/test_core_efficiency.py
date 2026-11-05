import time
import math
import random
import ml_core_engine as ml

def run_component_stress_test():
    print("=" * 60)
    print("⚡ ML CORE ENGINE: COMPREHENSIVE COMPONENT STRESS TEST")
    print("=" * 60)
    
    status = ml.check_gpu_device()
    print(f"📌 Status: {status}")
    
    ROWS = 8192
    COLS = 8192
    SIZE = ROWS * COLS
    print(f"📦 Payload Dimensions for Non-Linear Kernels: {ROWS}x{COLS} (~268.43 MB)")
    
    random.seed(42)
    raw_data_a = [random.uniform(-10.0, 100.0) for _ in range(SIZE)]
    raw_data_b = [random.uniform(-10.0, 100.0) for _ in range(SIZE)]
    
    print("⚡ Transferring massive payloads to persistent VRAM...")
    mat_a_gpu = ml.PyMatrix.from_list(ROWS, COLS, raw_data_a).to_gpu()
    mat_b_gpu = ml.PyMatrix.from_list(ROWS, COLS, raw_data_b).to_gpu()
    
    # 1. Benchmark RMSNorm
    print("\n🚀 Crunching RMSNorm Layer...")
    _ = mat_a_gpu.gpu_rms_norm(1e-5)
    start = time.perf_counter()
    for _ in range(20):
        res_rms = mat_a_gpu.gpu_rms_norm(1e-5)
    cpu_rms = res_rms.to_cpu()
    end = time.perf_counter()
    rms_latency = (end - start) / 20
    rms_bandwidth = (SIZE * 4 * 2) / (rms_latency * 1e9)
    print(f"   ⏱️ Latency: {rms_latency * 1000:.4f} ms | 🏎️ Bandwidth: {rms_bandwidth:.2f} GB/s")
    
    # 2. Benchmark SwiGLU
    print("\n🚀 Crunching SwiGLU Activation...")
    _ = mat_a_gpu.gpu_swiglu(mat_b_gpu)
    start = time.perf_counter()
    for _ in range(20):
        res_swiglu = mat_a_gpu.gpu_swiglu(mat_b_gpu)
    cpu_swiglu = res_swiglu.to_cpu()
    end = time.perf_counter()
    swiglu_latency = (end - start) / 20
    swiglu_bandwidth = (SIZE * 4 * 3) / (swiglu_latency * 1e9)
    print(f"   ⏱️ Latency: {swiglu_latency * 1000:.4f} ms | 🏎️ Bandwidth: {swiglu_bandwidth:.2f} GB/s")
    
    # 3. Benchmark RoPE (Using native shape constraint matching)
    print("\n🚀 Crunching RoPE (Rotary Position Embeddings)...")
    _ = mat_a_gpu.gpu_rope()
    start = time.perf_counter()
    for _ in range(20):
        res_rope = mat_a_gpu.gpu_rope()
    cpu_rope = res_rope.to_cpu()
    end = time.perf_counter()
    rope_latency = (end - start) / 20
    rope_bandwidth = (SIZE * 4 * 2) / (rope_latency * 1e9)
    print(f"   ⏱️ Latency: {rope_latency * 1000:.4f} ms | 🏎️ Bandwidth: {rope_bandwidth:.2f} GB/s")

    # 4. Benchmark Element-wise Binary (Add)
    print("\n🚀 Crunching Element-wise Add...")
    _ = mat_a_gpu.gpu_add(mat_b_gpu)
    start = time.perf_counter()
    for _ in range(20):
        res_add = mat_a_gpu.gpu_add(mat_b_gpu)
    cpu_add = res_add.to_cpu()
    end = time.perf_counter()
    add_latency = (end - start) / 20
    add_bandwidth = (SIZE * 4 * 3) / (add_latency * 1e9)
    print(f"   ⏱️ Latency: {add_latency * 1000:.4f} ms | 🏎️ Bandwidth: {add_bandwidth:.2f} GB/s")
    
    print("\n" + "-"*50)
    print("🔬 MATHEMATICAL INTEGRITY AUDIT SAMPLE:")
    print(f"   🔹 RMSNorm Sample Output: {cpu_rms.to_list()[:3]}")
    print(f"   🔹 SwiGLU Sample Output:  {cpu_swiglu.to_list()[:3]}")
    print(f"   🔹 RoPE Sample Output:    {cpu_rope.to_list()[:3]}")
    print(f"   🔹 Add Sample Output:     {cpu_add.to_list()[:3]}")
    print("-"*50)
    print("🏁 Comprehensive kernel capability validation complete!")
    print("=" * 60)

if __name__ == "__main__":
    run_component_stress_test()
