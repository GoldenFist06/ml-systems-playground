import time
import ml_core_engine as ml

def run_pure_flops_benchmark(dim):
    print("\n" + "=" * 70)
    print(f"🔬 PROFILING: Matrix Dimension {dim}x{dim} | FP32 Computations")
    print("=" * 70)
    
    a_cpu = ml.PyMatrix(dim, dim, 1.0001)
    b_cpu = ml.PyMatrix(dim, dim, 0.9999)
    
    a_gpu = a_cpu.to_gpu()
    b_gpu = b_cpu.to_gpu()
    
    for _ in range(5):
        warm_res = a_gpu.gpu_matmul(b_gpu, False)
        _ = warm_res.to_cpu()
    
    num_loops = 100
    print(f"🚀 Launching {num_loops} Continuous Asynchronous Dispatches to GPU...")
    
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
    
    total_elapsed = (end_time - start_time) - sync_overhead
    elapsed_per_loop = total_elapsed / num_loops
    
    gflops = (2.0 * (dim ** 3)) / (elapsed_per_loop * 1e9)
    
    print(f"⏱️ Total Time for {num_loops} Loops : {total_elapsed:.4f} seconds")
    print(f"⏱️ Pure Execution Latency/Loop  : {elapsed_per_loop:.5f} seconds")
    print(f"🚀 Absolute Sustained Throughput: ⚡ {gflops:.2f} GFLOPS")
    print("=" * 70)

def main():
    print("=" * 70)
    print("💎 ML CORE ENGINE - ABSOLUTE PURE HARDWARE FLOPS CAPTURE BENCHMARK")
    print("=" * 70)
    
    device_status = ml.check_gpu_device()
    print(f"📌 Hardware Device: {device_status}")
    
    dimensions = [4000, 6000, 7000, 8192]
    
    for dim in dimensions:
        run_pure_flops_benchmark(dim)
        time.sleep(2) 

    print("\n🏁 Pure hardware performance profiling successfully concluded!")

if __name__ == "__main__":
    main()
