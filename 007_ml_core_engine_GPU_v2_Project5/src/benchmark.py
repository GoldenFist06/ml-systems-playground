import time
from ml_core_engine import PyMatrix, check_gpu_device

def run_benchmark():
    print("--- 🚀 Starting GPU-Accelerated Performance Benchmark (Fused Kernel) ---")
    
    # Check GPU active state
    print(check_gpu_device())
    
    size = 500
    print(f"Creating matrices of size {size}x{size} for GPU compute...")
    
    # التصحيح هنا: استدعاء الكلاس مباشرة عبر __init__ (بدون .new)
    a = PyMatrix(size, size, 1.5)
    b = PyMatrix(size, size, 2.0)
    
    # 1. Benchmark standard GPU MatMul
    start_time = time.time()
    res_normal = a.gpu_matmul(b, use_relu=False)
    elapsed_normal = time.time() - start_time
    print(f"⚡ GPU MatMul ({size}x{size}) completed in: {elapsed_normal:.6f} seconds!")
    
    # 2. Benchmark Fused GPU MatMul + ReLU (النواة الجديدة الخارقة)
    start_time = time.time()
    res_fused = a.gpu_matmul(b, use_relu=True)
    elapsed_fused = time.time() - start_time
    print(f"🔥 Fused GPU MatMul + ReLU ({size}x{size}) completed in: {elapsed_fused:.6f} seconds!")
    
    print("--- Benchmark Completed Successfully! ---")

if __name__ == "__main__":
    run_benchmark()