import time
from ml_core_engine import PyMatrix, check_gpu_device

def run_heavy_stress_test():
    print("--- 🚀 Starting Extreme GPU Hardware Stress Test ---")
    print(check_gpu_device())
    
    sizes = [2000, 4000, 6000, 8000]
    iterations = 5
    
    for size in sizes:
        print(f"\n----------------------------------------")
        print(f"📦 Allocating massive matrices of size {size}x{size} ({size * size:,} elements each)...")
        
        a = PyMatrix(size, size, 1.05)
        b = PyMatrix(size, size, 2.02)
        
        print(f"🔥 Running continuous stress loop ({iterations} iterations for Fused MatMul + ReLU)...")
        
        start_time = time.time()
        for _ in range(iterations):
            _ = a.gpu_matmul(b, use_relu=True)
        elapsed_total = time.time() - start_time
        
        avg_time = elapsed_total / iterations
        gflops = ((2 * (size ** 3)) / avg_time) / 1e9
        
        print(f"   -> Total time for {iterations} runs: {elapsed_total:.4f} seconds")
        print(f"   -> Average time per run: {avg_time:.4f} seconds")
        print(f"   -> Sustained Performance: {gflops:.2f} GFLOPS")

    print("\n--- 🏁 Extreme Stress Test Completed Successfully! ---")

if __name__ == "__main__":
    run_heavy_stress_test()