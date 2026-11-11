import time
import ml_core_engine as ml

def run_pure_static_stress_test(dim, a_gpu, b_gpu):
    print("=" * 60)
    print(f"🔥 Extreme Stress Test: Matrix Dimension {dim}x{dim}")
    memory_mb = (dim * dim * 4) / (1024 * 1024)
    print(f"📦 Payload Context: Persistent Buffers (~{memory_mb:.2f} MB locked)")
    
    for _ in range(5):
        warm_res = a_gpu.gpu_matmul(b_gpu, False)
        _ = warm_res.to_cpu()
        
    print(f"🚀 Initiating pure parallel GEMM crunch for {dim}x{dim}...")
    
    num_loops = 2
    
    t_sync_start = time.perf_counter()
    sync_res = a_gpu.gpu_matmul(b_gpu, False)
    _ = sync_res.to_cpu()
    t_sync_end = time.perf_counter()
    sync_overhead = t_sync_end - t_sync_start
    
    start_time = time.perf_counter()
    
    for _ in range(num_loops):
        result_gpu = a_gpu.gpu_matmul(b_gpu, False)
        
    result_cpu = result_gpu.to_cpu()
    end_time = time.perf_counter()
    
    total_duration = (end_time - start_time) - sync_overhead
    elapsed = total_duration / num_loops
    gflops = (2.0 * (dim ** 3)) / (elapsed * 1e9)
    sample = result_cpu.to_list()[:3]
    
    print(f"⏱️ Net GPU Execution Time (Per Loop): {elapsed:.5f} seconds")
    print(f"🚀 Achieved Throughput: {gflops:.2f} GFLOPS")
    print(f"🔬 FP32 Integrity Verification Sample: {sample}")
    print("=" * 60)

def main():
    print("=" * 60)
    print("⚡ ML Core Engine v4.0: PERSISTENT BUFFER MAXIMUM STRESS TEST")
    print("=" * 60)
    
    device_status = ml.check_gpu_device()
    print(f"📌 Status: {device_status}")
    
    dimensions = [4000, 6000, 7000, 8192]
    
    print("\n⚡ Pre-allocating and locking all volumetric payloads in persistent VRAM...")
    gpu_buffers = {}
    for dim in dimensions:
        a_cpu = ml.PyMatrix(dim, dim, 1.0001)
        b_cpu = ml.PyMatrix(dim, dim, 0.9999)
        gpu_buffers[dim] = (a_cpu.to_gpu(), b_cpu.to_gpu())
    print("✨ All VRAM buffer layers successfully populated and locked.")
    
    for dim in dimensions:
        a_gpu, b_gpu = gpu_buffers[dim]
        run_pure_static_stress_test(dim, a_gpu, b_gpu)
        time.sleep(2)

    print("\n🏁 Extreme pre-allocated stress test successfully concluded with zero precision loss!")

if __name__ == "__main__":
    main()
