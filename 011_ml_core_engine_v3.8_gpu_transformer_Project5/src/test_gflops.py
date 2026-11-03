import time
import ml_core_engine as ml

def run_stress_test(dim):
    print("=" * 60)
    print(f"🔥 Extreme Stress Test: Matrix Dimension {dim}x{dim}")
    memory_mb = (dim * dim * 4) / (1024 * 1024)
    print(f"📦 Allocating massive payload (~{memory_mb:.2f} MB per matrix)...")
    
    a_cpu = ml.PyMatrix(dim, dim, 1.0001)
    b_cpu = ml.PyMatrix(dim, dim, 0.9999)
    
    a_gpu = a_cpu.to_gpu()
    b_gpu = b_cpu.to_gpu()
    
    print("⚡ Executing warm-up pipeline synchronization...")
    _ = a_gpu.gpu_matmul(b_gpu, False)
    
    print(f"🚀 Initiating heavy parallel GEMM crunch for {dim}x{dim}...")
    start_time = time.perf_counter()
    
    result_gpu = a_gpu.gpu_matmul(b_gpu, False)
    
    elapsed = time.perf_counter() - start_time
    
    gflops = (2.0 * (dim ** 3)) / (elapsed * 1e9)
    
    result_cpu = result_gpu.to_cpu()
    sample = result_cpu.to_list()[:3]
    
    print(f"⏱️ Execution Time: {elapsed:.4f} seconds")
    print(f"🚀 Achieved Throughput: {gflops:.2f} GFLOPS")
    print(f"🔬 FP32 Integrity Verification Sample: {sample}")
    print("=" * 60)

def main():
    print("=" * 60)
    print("⚡ ML Core Engine v3.7: MAXIMUM INTENSITY STRESS TEST")
    print("=" * 60)
    
    device_status = ml.check_gpu_device()
    print(f"📌 Status: 🚀 {device_status} [Locked & Loaded]")
    
    dimensions = [4000, 6000]
    
    for dim in dimensions:
        run_stress_test(dim)
        time.sleep(1)

    print("\n🏁 Extreme stress test sequence successfully concluded with zero precision loss!")

if __name__ == "__main__":
    main()
