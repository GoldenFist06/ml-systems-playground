import time
import ml_core_engine as ml

def profile_engine_overhead():
    status = ml.check_gpu_device()
    print(status)

    
    _ = ml.PyMatrix(4, 4, 1.0).to_gpu().gpu_matmul(ml.PyMatrix(4, 4, 1.0).to_gpu(), False)

    micro_a = ml.PyMatrix(4, 4, 1.0).to_gpu()
    micro_b = ml.PyMatrix(4, 4, 1.0).to_gpu()

    NUM_RUNS = 100
    start_null = time.perf_counter()
    for _ in range(NUM_RUNS):
        _ = micro_a.gpu_matmul(micro_b, False)
    end_null = time.perf_counter()
    null_dispatch_time = (end_null - start_null) / NUM_RUNS

    
    DIM = 4000
    large_a = ml.PyMatrix(DIM, DIM, 1.0).to_gpu()
    large_b = ml.PyMatrix(DIM, DIM, 1.0).to_gpu()

    start_large = time.perf_counter()
    _ = large_a.gpu_matmul(large_b, False)
    end_large = time.perf_counter()
    total_large_time = end_large - start_large

    pure_gpu_time = max(0.0, total_large_time - null_dispatch_time)
    overhead_percentage = (null_dispatch_time / total_large_time) * 100

    print("--------------------------------------------------")
    print(f"📊 Engine Overhead Profiling Results:")
    print(f"⏱️ Raw Dispatch Overhead (Null Run): {null_dispatch_time*1000:.4f} ms")
    print(f"⏱️ Total Execution Time ({DIM}x{DIM}): {total_large_time*1000:.4f} ms")
    print(f"⏱️ Pure GPU Hardware Compute Time: {pure_gpu_time*1000:.4f} ms")
    print(f"🔥 Core Scheduling Overhead Ratio: {overhead_percentage:.2f}%")
    print("--------------------------------------------------")

if __name__ == "__main__":
    profile_engine_overhead()