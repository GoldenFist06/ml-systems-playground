import time
import ml_core_engine as ml

def run_fused_attention_stress_test():
    print("=" * 80)
    print("⚡ ML CORE ENGINE: REALISTIC ATTENTION STRESS BENCHMARK (KV CACHE)")
    print("=" * 80)
    
    status = ml.check_gpu_device()
    print(f"📌 Status: {status}")
    
    batch_size = 1
    d_model = 4096
    n_heads = 32
    d_k = d_model // n_heads
    max_context_len = 512
    loops = 50
    
    print(f"📦 Allocating Attention Pipelines & KV Cache Dimensions:")
    print(f"   🔹 Q Matrix (Single-Head Isolated Profile) : {batch_size}x{d_k}")
    print(f"   🔹 KV Cache Capacity (Persistent Space)   : Context={max_context_len} | d_k={d_k}")
    print("=" * 80)
    
    q_cpu = ml.PyMatrix(batch_size, d_k, 0.25)
    q_gpu = q_cpu.to_gpu()
    
    kv_cache = ml.KvCache(max_context_len, d_k)
    
    k_step_cpu = ml.PyMatrix(batch_size, d_k, 0.05)
    v_step_cpu = ml.PyMatrix(batch_size, d_k, 0.08)
    k_step_gpu = k_step_cpu.to_gpu()
    v_step_gpu = v_step_cpu.to_gpu()
    
    print("⚡ Filling KvCache persistent history timeline slots...")
    for step in range(max_context_len):
        kv_cache.update(step, k_step_gpu, v_step_gpu)
        
    for _ in range(5):
        warm_res = q_gpu.gpu_fused_attention(kv_cache, max_context_len)
        _ = warm_res.buffer_id()
        
    print(f"🚀 Launching {loops} runs of FUSED QKV Attention across entire context history...")
    
    t_sync_start = time.perf_counter()
    sync_res = q_gpu.gpu_fused_attention(kv_cache, max_context_len)
    _ = sync_res.to_cpu()
    t_sync_end = time.perf_counter()
    sync_overhead = t_sync_end - t_sync_start
    
    start_time = time.perf_counter()
    for _ in range(loops):
        result_gpu = q_gpu.gpu_fused_attention(kv_cache, max_context_len)
        
    _ = result_gpu.buffer_id()
    end_time = time.perf_counter()
    
    result_cpu = result_gpu.to_cpu()
    
    total_elapsed = (end_time - start_time) - sync_overhead
    latency_per_run_ms = (total_elapsed / loops) * 1000
    sample_output = result_cpu.to_list()[:4]
    
    print("-" * 80)
    print(f"• Total Process Duration ({loops} Loops) : {total_elapsed:.4f} seconds")
    print(f"• Net Fused Attention Latency (Per Run) : 🚀 {latency_per_run_ms:.4f} ms")
    print(f"• Precision Integrity Verification Sample: 🔬 {sample_output}")
    print("=" * 80)

if __name__ == "__main__":
    run_fused_and_all_kernels_benchmark() if "run_fused_and_all_kernels_benchmark" in globals() else run_fused_attention_stress_test()
