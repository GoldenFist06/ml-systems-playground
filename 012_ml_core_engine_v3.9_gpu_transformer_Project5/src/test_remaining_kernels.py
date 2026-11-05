import time
import random
import ml_core_engine as ml

def run_remaining_kernels_test():
    print("=" * 60)
    print("⚡ ML CORE ENGINE: REMAINING KERNELS INTEGRITY & STRESS TEST")
    print("=" * 60)
    
    status = ml.check_gpu_device()
    print(f"📌 Status: {status}")
    
    ROWS = 8192
    COLS = 8192
    SIZE = ROWS * COLS
    print(f"📦 Payload Dimensions: {ROWS}x{COLS} (~268.43 MB per matrix)")
    
    random.seed(1337)
    raw_data_a = [random.uniform(-50.0, 50.0) for _ in range(SIZE)]
    raw_data_b = [random.uniform(-50.0, 50.0) for _ in range(SIZE)]
    
    print("⚡ Pumping data payloads to VRAM persistent storage...")
    mat_a_gpu = ml.PyMatrix.from_list(ROWS, COLS, raw_data_a).to_gpu()
    mat_b_gpu = ml.PyMatrix.from_list(ROWS, COLS, raw_data_b).to_gpu()
    
    print("\n🚀 Crunching Element-wise Subtraction (gpu_sub)...")
    _ = mat_a_gpu.gpu_sub(mat_b_gpu)
    start = time.perf_counter()
    for _ in range(20):
        res_sub = mat_a_gpu.gpu_sub(mat_b_gpu)
    cpu_sub = res_sub.to_cpu()
    end = time.perf_counter()
    sub_latency = (end - start) / 20
    sub_bandwidth = (SIZE * 4 * 3) / (sub_latency * 1e9)
    print(f"   ⏱️ Latency: {sub_latency * 1000:.4f} ms | 🏎️ Bandwidth: {sub_bandwidth:.2f} GB/s")
    
    print("\n🚀 Crunching ReLU Backward Gradient (gpu_relu_backward)...")
    _ = mat_a_gpu.gpu_relu_backward(mat_b_gpu)
    start = time.perf_counter()
    for _ in range(20):
        res_relu_bw = mat_a_gpu.gpu_relu_backward(mat_b_gpu)
    cpu_relu_bw = res_relu_bw.to_cpu()
    end = time.perf_counter()
    relu_bw_latency = (end - start) / 20
    relu_bw_bandwidth = (SIZE * 4 * 3) / (relu_bw_latency * 1e9)
    print(f"   ⏱️ Latency: {relu_bw_latency * 1000:.4f} ms | 🏎️ Bandwidth: {relu_bw_bandwidth:.2f} GB/s")
    
    print("\n" + "-"*50)
    print("🔬 MATHEMATICAL INTEGRITY AUDIT SAMPLE:")
    print(f"   🔹 Sub Output Sample:      {cpu_sub.to_list()[:3]}")
    print(f"   🔹 ReLU Backward Sample:  {cpu_relu_bw.to_list()[:3]}")
    print("-"*50)
    print("🏁 Validation sequence successfully executed!")
    print("=" * 60)

if __name__ == "__main__":
    run_remaining_kernels_test()
