import time
import math
import random

try:
    import ml_core_engine as ml
except ImportError:
    exit(1)


def generate_random_matrix(rows, cols):
    data = [random.uniform(-0.1, 0.1) for _ in range(rows * cols)]
    return ml.PyMatrix.from_list(rows, cols, data)


def run_benchmark():
    status = ml.check_gpu_device()
    print(status)

    B = 1
    SEQ_LEN = 4000
    HEAD_DIM = 4000

    q_cpu = generate_random_matrix(B, HEAD_DIM)
    k_cpu = generate_random_matrix(SEQ_LEN, HEAD_DIM)
    v_cpu = generate_random_matrix(SEQ_LEN, HEAD_DIM)

    start_upload = time.perf_counter()
    q_gpu = q_cpu.to_gpu()
    
    
    kv_cache = ml.KvCache(SEQ_LEN, HEAD_DIM)
    
    k_list = k_cpu.to_list()
    v_list = v_cpu.to_list()
    
   
    for i in range(SEQ_LEN):
        single_k = ml.PyMatrix.from_list(1, HEAD_DIM, k_list[i * HEAD_DIM : (i + 1) * HEAD_DIM]).to_gpu()
        single_v = ml.PyMatrix.from_list(1, HEAD_DIM, v_list[i * HEAD_DIM : (i + 1) * HEAD_DIM]).to_gpu()
        kv_cache.update(i, single_k, single_v)
        
    end_upload = time.perf_counter()
    print(f"Upload time: {(end_upload - start_upload)*1000:.2f} ms")

    
    _ = q_gpu.gpu_attention(kv_cache, SEQ_LEN)

    NUM_RUNS = 10
    total_time = 0.0

    for i in range(NUM_RUNS):
        start_compute = time.perf_counter()
        result_gpu = q_gpu.gpu_attention(kv_cache, SEQ_LEN)
        end_compute = time.perf_counter()
        total_time += end_compute - start_compute

    avg_time = total_time / NUM_RUNS

    total_ops = 2 * B * SEQ_LEN * HEAD_DIM + 2 * B * SEQ_LEN * HEAD_DIM
    gflops = (total_ops / avg_time) / 1e9

    print(f"Average Compute Time: {avg_time*1000:.4f} ms")
    print(f"Total Operations: {total_ops}")
    print(f"Performance: {gflops:.2f} GFLOPS")

    start_download = time.perf_counter()
    result_cpu = result_gpu.to_cpu()
    end_download = time.perf_counter()
    print(f"Download time: {(end_download - start_download)*1000:.2f} ms")


if __name__ == "__main__":
    random.seed(42)
    run_benchmark()