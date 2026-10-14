import time
import ml_core_engine

def run_benchmark():
    print("--- 🚀 Starting GPU-Accelerated Performance Benchmark ---")
    
    size = 500
    print(f"Creating matrices of size {size}x{size} for GPU compute...")
    
    a = ml_core_engine.PyMatrix(size, size, 1.5)
    b = ml_core_engine.PyMatrix(size, size, 0.5)
    
    # Benchmark GPU Matrix Multiplication
    start_time = time.time()
    result = a.gpu_matmul(b)
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"⚡ GPU MatMul ({size}x{size}) completed in: {elapsed:.6f} seconds!")
    
    # Benchmark GPU ReLU Activation
    start_time = time.time()
    for _ in range(1000):
        _ = a.gpu_relu()
    end_time = time.time()
    
    relu_elapsed = end_time - start_time
    print(f"⚡ 1,000 GPU ReLU iterations completed in: {relu_elapsed:.6f} seconds!")

if __name__ == "__main__":
    run_benchmark()