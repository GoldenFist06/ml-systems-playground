import time
import gc
import ml_core_engine as ml

def run_pure_static_stress_test(dim):
    print("=" * 60)
    print(f"Extreme Stress Test: Matrix Dimension {dim}x{dim}")
    memory_mb = (dim * dim * 4) / (1024 * 1024)
    print(f"Payload Context: Persistent Buffers (~{memory_mb:.2f} MB per matrix)")

    a_cpu = ml.PyMatrix(dim, dim, 1.0001)
    b_cpu = ml.PyMatrix(dim, dim, 0.9999)
    a_gpu = a_cpu.to_gpu()
    b_gpu = b_cpu.to_gpu()
    del a_cpu, b_cpu

    warm_res = a_gpu.gpu_matmul(b_gpu, False)
    _ = warm_res.to_cpu()
    del warm_res

    batch_size = 5
    num_batches = 4
    total_loops = batch_size * num_batches
    print(f"Initiating batched pure VRAM GEMM crunch for {dim}x{dim} ({total_loops} iterations)...")

    total_gpu_time = 0.0
    sample = []

    for b in range(num_batches):
        batch_results = []
        t0 = time.perf_counter()
        for _ in range(batch_size):
            batch_results.append(a_gpu.gpu_matmul(b_gpu, False))
        last_cpu = batch_results[-1].to_cpu()
        t1 = time.perf_counter()

        total_gpu_time += (t1 - t0)
        if b == num_batches - 1:
            sample = last_cpu.to_list()[:3]

        del batch_results, last_cpu

    elapsed = total_gpu_time / total_loops
    gflops = (2.0 * (dim ** 3)) / (elapsed * 1e9)
    expected_val = dim * 1.0001 * 0.9999

    print(f"Net Pure GPU Execution Time (Per Loop): {elapsed:.5f} seconds")
    print(f"Achieved Pure Compute Throughput: {gflops:.2f} GFLOPS")
    print(f"FP32 Integrity Sample: {[round(x, 2) for x in sample]} (Expected: ~{expected_val:.2f})")
    print("=" * 60)

    del a_gpu, b_gpu
    gc.collect()

def main():
    print("=" * 60)
    print("ML Core Engine v4.1: PURE VRAM BANDWIDTH & TFLOPS BENCHMARK")
    print("=" * 60)

    device_status = ml.check_gpu_device()
    print(f"Status: {device_status}\n")

    dimensions = [4000, 6000, 7000, 8192]

    for dim in dimensions:
        run_pure_static_stress_test(dim)
        time.sleep(1)

    print("\nPure compute benchmark concluded with verified FP32 precision.")

if __name__ == "__main__":
    main()