import time
import random
import ml_core_engine as ml

print("=" * 70)
print("ML CORE ENGINE PERFORMANCE BENCHMARK")
print("=" * 70)

sizes = [
    1024,
    2048,
    4000,
]

for n in sizes:

    print()
    print("=" * 70)
    print(f"SIZE = {n} x {n}")
    print("=" * 70)

    a_data = [
        random.uniform(-1.0, 1.0)
        for _ in range(n * n)
    ]

    b_data = [
        random.uniform(-1.0, 1.0)
        for _ in range(n * n)
    ]

    a = ml.PyMatrix.from_list(n, n, a_data)
    b = ml.PyMatrix.from_list(n, n, b_data)

    a_gpu = a.to_gpu()
    b_gpu = b.to_gpu()

    print("\nGPU MATMUL WARMUP...")
    _ = a_gpu.gpu_matmul(b_gpu, False)
    print("DONE")

    times = []
    print("\nGPU MATMUL RUNS")

    for run in range(5):
        start = time.perf_counter()
        c = a_gpu.gpu_matmul(b_gpu, False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Run {run+1}: {elapsed:.6f} sec")

    avg_time = sum(times) / len(times)
    flops = 2 * n * n * n
    gflops = flops / avg_time / 1e9

    print()
    print(f"Average Time : {avg_time:.6f} sec")
    print(f"Estimated GFLOPS : {gflops:.2f}")

    start = time.perf_counter()
    identity = ml.PyMatrix.from_list(n, n, [1.0 if i == j else 0.0 for i in range(n) for j in range(n)]).to_gpu()
    _ = a_gpu.gpu_matmul(identity, use_relu=True)
    relu_time = time.perf_counter() - start
    print(f"gpu_relu      : {relu_time:.6f} sec")

    start = time.perf_counter()
    _ = a_gpu.gpu_add(b_gpu)
    add_time = time.perf_counter() - start
    print(f"gpu_add       : {add_time:.6f} sec")

    start = time.perf_counter()
    _ = a_gpu.gpu_scale(0.5)
    scale_time = time.perf_counter() - start
    print(f"gpu_scale     : {scale_time:.6f} sec")

    start = time.perf_counter()
    _ = a_gpu.gpu_transpose()
    transpose_time = time.perf_counter() - start
    print(f"gpu_transpose : {transpose_time:.6f} sec")

print()
print("=" * 70)
print("BENCHMARK FINISHED")
print("=" * 70)
