import time
import ml_core_engine

def run_benchmark():
    print("--- 🚀 Starting Rust Core Performance Benchmark ---")
    
   
    size = 500
    print(f"Creating matrices of size {size}x{size}...")
    
    a = ml_core_engine.PyMatrix(size, size, 1.5)
    b = ml_core_engine.PyMatrix(size, size, 0.5)
    
    
    start_time = time.time()
    result = a.matmul(b)
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"✅ MatMul ({size}x{size}) completed in: {elapsed:.6f} seconds!")
    
   
    start_time = time.time()
    for _ in range(1000):
        _ = a.relu()
    end_time = time.time()
    
    relu_elapsed = end_time - start_time
    print(f"✅ 1,000 ReLU iterations completed in: {relu_elapsed:.6f} seconds!")

if __name__ == "__main__":
    run_benchmark()