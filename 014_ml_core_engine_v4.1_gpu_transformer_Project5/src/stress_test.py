import time
import gc
import ml_core_engine as ml

def run_pure_flops_benchmark(dim):
    print("\n" + "=" * 70)
    print(f"🔬 PROFILING: Matrix Dimension {dim}x{dim} | FP32 Computations")
    print("=" * 70)
    
    a_cpu = ml.PyMatrix(dim, dim, 1.0001)
    b_cpu = ml.PyMatrix(dim, dim, 0.9999)
    
    a_gpu = a_cpu.to_gpu()
    b_gpu = b_cpu.to_gpu()
    del a_cpu, b_cpu
    
    for _ in range(2):
        warm_res = a_gpu.gpu_matmul(b_gpu, False)
        _ = warm_res.to_cpu()
        del warm_res
    
    num_loops = 20 if dim >= 7000 else 50
    batch_size = 5
    num_batches = num_loops // batch_size
    total_iterations = num_batches * batch_size
    
    print(f"🚀 Launching batched execution ({total_iterations} iterations, batch size {batch_size})...")
    
    start_time = time.perf_counter()
    last_cpu = None
    
    for b in range(num_batches):
        batch_results = []
        for _ in range(batch_size):
            batch_results.append(a_gpu.gpu_matmul(b_gpu, False))
        last_cpu = batch_results[-1].to_cpu()
        del batch_results
        
    end_time = time.perf_counter()
    
    total_elapsed = end_time - start_time
    elapsed_per_loop = total_elapsed / total_iterations
    
    gflops = (2.0 * (dim ** 3)) / (elapsed_per_loop * 1e9)
    expected_val = dim * 1.0001 * 0.9999
    sample = last_cpu.to_list()[:3]
    
    print(f"⏱️ Total Time for {total_iterations} Loops : {total_elapsed:.4f} seconds")
    print(f"⏱️ Pure Execution Latency/Loop  : {elapsed_per_loop:.5f} seconds")
    print(f"🚀 Absolute Sustained Throughput: ⚡ {gflops:.2f} GFLOPS")
    print(f"🔬 FP32 Verification Sample     : {[round(x, 2) for x in sample]} (Expected: ~{expected_val:.2f})")
    print("=" * 70)
    
    del a_gpu, b_gpu, last_cpu
    gc.collect()

def main():
    print("=" * 70)
    print(" ML CORE ENGINE - ABSOLUTE PURE HARDWARE FLOPS CAPTURE BENCHMARK")
    print("=" * 70)
    
    device_status = ml.check_gpu_device()
    print(f"📌 Hardware Device: {device_status}")
    
    dimensions = [4000, 6000, 7000, 8192]
    
    for dim in dimensions:
        run_pure_flops_benchmark(dim)
        time.sleep(1) 

    print("\n🏁 Pure hardware performance profiling successfully concluded!")

if __name__ == "__main__":
    main()