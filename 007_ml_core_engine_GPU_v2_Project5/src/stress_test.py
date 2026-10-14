import time
from ml_core_engine import PyMatrix, check_gpu_device

def run_stress_test():
    print("--- 🚀 Starting GPU Hardware Stress Test (Massive Matrices) ---")
    

    print(check_gpu_device())
    
    
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        print(f"\n----------------------------------------")
        print(f"📦 Allocating matrices of size {size}x{size} ({size * size:,} elements each)...")
        
     
        a = PyMatrix(size, size, 1.05)
        b = PyMatrix(size, size, 2.02)
        
      
        print(f"⚡ Running standard GPU MatMul...")
        start_time = time.time()
        _ = a.gpu_matmul(b, use_relu=False)
        elapsed_normal = time.time() - start_time
        print(f"   -> Completed in: {elapsed_normal:.4f} seconds")
        
        
        print(f"🔥 Running Fused GPU MatMul + ReLU...")
        start_time = time.time()
        _ = a.gpu_matmul(b, use_relu=True)
        elapsed_fused = time.time() - start_time
        print(f"   -> Completed in: {elapsed_fused:.4f} seconds")
        
     
        operations = 2 * (size ** 3)
        gflops = (operations / elapsed_fused) / 1e9
        print(f"💎 Estimated Performance: {gflops:.2f} GFLOPS")

    print("\n--- 🏁 Stress Test Completed Successfully! ---")

if __name__ == "__main__":
    run_stress_test()