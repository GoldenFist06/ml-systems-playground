import time
import ml_core_engine as ml

def benchmark_fused_attention():
    print("=" * 60)
    print("⚡ ML CORE ENGINE: REALISTIC ATTENTION STRESS BENCHMARK (KV CACHE)")
    print("=" * 60)
    
    status = ml.check_gpu_device()
    print(f"📌 Status: {status}")
    
    B = 1       
    SEQ_LEN = 4096 
    D_K = 4096     
    
    print(f"📦 Allocating Attention Pipelines & KV Cache:")
    print(f"   🔹 Q Matrix: {B}x{D_K}")
    print(f"   🔹 KV Cache Max Context: {SEQ_LEN}x{D_K}")
    
    q_cpu = ml.PyMatrix(B, D_K, 0.05)
    k_cpu = ml.PyMatrix(SEQ_LEN, D_K, 0.02)
    v_cpu = ml.PyMatrix(SEQ_LEN, D_K, 0.1)
    
    q_gpu = q_cpu.to_gpu()
    
    
    kv_cache = ml.KvCache(SEQ_LEN, D_K)
    
    print("⚡ Pumping payloads into VRAM persistent KvCache layers...")
    k_list = k_cpu.to_list()
    v_list = v_cpu.to_list()
    
   
    for i in range(SEQ_LEN):
        single_k = ml.PyMatrix.from_list(1, D_K, k_list[i * D_K : (i + 1) * D_K]).to_gpu()
        single_v = ml.PyMatrix.from_list(1, D_K, v_list[i * D_K : (i + 1) * D_K]).to_gpu()
        kv_cache.update(i, single_k, single_v)
    
    
    _ = q_gpu.gpu_fused_attention(kv_cache, SEQ_LEN)
    
    NUM_RUNS = 50
    print(f"🚀 Launching {NUM_RUNS} runs of FUSED QKV Attention with KvCache...")
    
    start = time.perf_counter()
    for _ in range(NUM_RUNS):
        res_gpu = q_gpu.gpu_fused_attention(kv_cache, SEQ_LEN)
        
    res_cpu = res_gpu.to_cpu()
    end = time.perf_counter()
    
    avg_latency = (end - start) / NUM_RUNS
    print(f"⏱️ Net Fused Attention Latency (Per Run): {avg_latency * 1000:.4f} ms")
    
    sample = res_cpu.to_list()[:4]
    print(f"🔬 Precision Integrity Check Sample: {sample}")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_fused_attention()