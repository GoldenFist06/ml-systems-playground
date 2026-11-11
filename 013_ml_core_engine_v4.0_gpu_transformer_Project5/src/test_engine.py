import time
import random
import ml_core_engine as ml

def run_benchmark():
    status = ml.check_gpu_device()
    print(status)

    B = 1
    SEQ_LEN = 4000
    HEAD_DIM = 4096
    NUM_RUNS = 20

    k_full_cpu = ml.PyMatrix(SEQ_LEN, HEAD_DIM, 0.01)
    v_full_cpu = ml.PyMatrix(SEQ_LEN, HEAD_DIM, 0.02)
    q_cpu = ml.PyMatrix(B, HEAD_DIM, 0.05)

    start_upload = time.perf_counter()
    q_gpu = q_cpu.to_gpu()
    k_full_gpu = k_full_cpu.to_gpu()
    v_full_gpu = v_full_cpu.to_gpu()
    kv_cache = ml.KvCache(SEQ_LEN, HEAD_DIM)
    
    kv_cache.update(0, k_full_gpu, v_full_gpu)
    end_upload = time.perf_counter()
    print(f"Upload time: {(end_upload - start_upload)*1000:.2f} ms")

    for _ in range(5):
        warm_res = q_gpu.gpu_attention(kv_cache, SEQ_LEN)
        _ = warm_res.buffer_id()

    start_time = time.perf_counter()
    for _ in range(NUM_RUNS):
        result_gpu = q_gpu.gpu_attention(kv_cache, SEQ_LEN)
        
    if isinstance(result_gpu, ml.GpuMatrix):
        _ = result_gpu.buffer_id()
    end_time = time.perf_counter()

    result_cpu = result_gpu.to_cpu()
    
    total_elapsed = end_time - start_time
    avg_time = total_elapsed / NUM_RUNS
    
    total_ops = 2 * B * SEQ_LEN * HEAD_DIM + 2 * B * SEQ_LEN * HEAD_DIM
    gflops = (total_ops / avg_time) / 1e9

    print(f"Average Compute Time: {avg_time*1000:.4f} ms")
    print(f"Total Operations: {total_ops}")
    print(f"Performance: {gflops:.2f} GFLOPS")

    start_download = time.perf_counter()
    _ = result_gpu.to_cpu()
    end_download = time.perf_counter()
    print(f"Download time: {(end_download - start_download)*1000:.2f} ms")

if __name__ == "__main__":
    random.seed(42)
    run_benchmark()
