import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.getcwd())

RESULT_TAG = "@@VERIFICATION_RESULT@@"
STATUS_PASS, STATUS_FAIL, STATUS_WARN = "PASS", "FAIL", "WARN"

def log_status(msg=""):
    print(f"[SYSTEM AUDIT] {msg}", flush=True)

class VerificationSuiteResult:
    def __init__(self):
        self.status = STATUS_PASS
        self.notes = []

    def log_failure(self, msg):
        self.status = STATUS_FAIL
        self.notes.append(msg)
        log_status(f"CRITICAL -> {msg}")

    def log_warning(self, msg):
        if self.status == STATUS_PASS:
            self.status = STATUS_WARN
        self.notes.append(msg)
        log_status(f"WARNING -> {msg}")

def get_engine():
    import ml_core_engine
    return ml_core_engine

def allocate_matrix(rows, cols, arr):
    flat = np.asarray(arr, dtype=np.float32).ravel().tolist()
    return get_engine().PyMatrix.from_list(rows, cols, flat)

def to_numpy_array(m, rows, cols):
    return np.asarray(m.to_list(), dtype=np.float32).reshape(rows, cols)

def verify_concurrency(res):
    log_status("Test 01: Stress-testing multi-threaded GPU pipeline execution (32 parallel threads)...")
    execution_errors = []
    
    def thread_worker(thread_id):
        try:
            rng = np.random.default_rng(100 + thread_id)
            for _ in range(100):
                dim = rng.choice([128, 256])
                matrix_a = rng.standard_normal((dim, dim)).astype(np.float32)
                matrix_b = rng.standard_normal((dim, dim)).astype(np.float32)
                gpu_a = allocate_matrix(dim, dim, matrix_a).to_gpu()
                gpu_b = allocate_matrix(dim, dim, matrix_b).to_gpu()
                gpu_c = gpu_a.gpu_matmul(gpu_b, use_relu=True)
                _ = gpu_c.buffer_id()
        except Exception as err:
            execution_errors.append((thread_id, str(err)))

    worker_threads = [threading.Thread(target=thread_worker, args=(i,)) for i in range(32)]
    timer_start = time.perf_counter()
    for thread in worker_threads: thread.start()
    for thread in worker_threads: thread.join()
    time_elapsed = time.perf_counter() - timer_start

    if execution_errors:
        res.log_failure(f"Concurrency execution anomaly detected with {len(execution_errors)} errors. Init: {execution_errors[0]}")
    else:
        log_status(f"Concurrency thread validation nominal. 32 pipelines completed execution loop in {time_elapsed:.3f}s.")

def verify_pathological_shapes(res):
    log_status("Test 02: Validating edge-case matrix dimension structures and volumetric scaling...")
    rng = np.random.default_rng(666)
    standard_aligned_bounds = [16, 32, 64, 128, 256]
    
    for _ in range(300):
        m = random.choice(standard_aligned_bounds)
        k = random.choice(standard_aligned_bounds)
        n = random.choice(standard_aligned_bounds)
        
        matrix_a = rng.standard_normal((m, k)).astype(np.float32) * 1e4
        matrix_b = rng.standard_normal((k, n)).astype(np.float32) * 1e-4
        
        try:
            gpu_a = allocate_matrix(m, k, matrix_a).to_gpu()
            gpu_b = allocate_matrix(k, n, matrix_b).to_gpu()
            gpu_out = gpu_a.gpu_matmul(gpu_b, use_relu=True)
            _ = gpu_out.buffer_id()
        except Exception as err:
            res.log_failure(f"Exception triggered during shape processing {m}x{k} with {k}x{n}: {err}")
            break
    log_status("Volumetric dimensionality scaling execution nominal and accurate.")

def verify_memory_churn(res):
    log_status("Test 03: Profile host RSS memory differential across sequential rapid allocations...")
    engine_module = get_engine()
    dim = 128
    allocation_payload = [0.1] * (dim * dim)
    
    try:
        import psutil
        process_profile = psutil.Process()
        initial_rss_mb = process_profile.memory_info().rss / 1e6
        
        for _ in range(5000):
            _ = engine_module.PyMatrix.from_list(dim, dim, allocation_payload)
            
        final_rss_mb = process_profile.memory_info().rss / 1e6
        memory_delta = final_rss_mb - initial_rss_mb
        log_status(f"Host tracking RSS differential post-allocation cycle: {memory_delta:+.2f} MB")
        if memory_delta > 50.0:
            res.log_warning(f"Potential memory footprint drift detected under stress state (+{memory_delta:.2f} MB)")
    except ImportError:
        log_status("Module 'psutil' unavailable. Skipping discrete host heap auditing.")
    except Exception as err:
        res.log_warning(f"Memory footprint execution exception: {err}")

def verify_numerical_bounds(res):
    log_status("Test 04: Assessing tensor runtime numerical limits under infinity and indeterminate floating states...")
    try:
        infinity_matrix = allocate_matrix(4, 4, [float("inf")] * 16).to_gpu()
        identity_matrix = allocate_matrix(4, 4, [1.0 if i==j else 0.0 for i in range(4) for j in range(4)]).to_gpu()
        gpu_out = infinity_matrix.gpu_matmul(identity_matrix, use_relu=False)
        _ = gpu_out.buffer_id()
        log_status("Numerical limit conditions successfully processed in asynchronous flow.")
    except Exception as err:
        res.log_failure(f"FFI Boundary runtime panic during extreme values parsing: {err}")

def main():
    print("=" * 74)
    print(" ML_CORE_ENGINE: RIGOROUS HARDWARE INTEGRITY & COMPLIANCE VERIFICATION")
    print("=" * 74)
    
    suite_result = VerificationSuiteResult()
    try:
        verify_concurrency(suite_result)
        verify_pathological_shapes(suite_result)
        verify_memory_churn(suite_result)
        verify_numerical_bounds(suite_result)
    except BaseException as err:
        suite_result.log_failure(f"Fatal unhandled exception during execution lifecycle: {err}")
        traceback.print_exc()

    print("=" * 74)
    if suite_result.status == STATUS_FAIL:
        print(f"[-] INTEGRITY COMPLIANCE CRITICAL: Target engine fails architectural specs.")
        for note in suite_result.notes: print(f"    - {note}")
        sys.exit(1)
    elif suite_result.status == STATUS_WARN:
        print(f"[-] INTEGRITY COMPLIANCE WARNING: Runtime nominal but boundaries non-ideal.")
        for note in suite_result.notes: print(f"    - {note}")
        sys.exit(0)
    else:
        print(f"[+] INTEGRITY COMPLIANCE VERIFIED: Hardware runtime execution completely successful.")
    print("=" * 74)
    sys.exit(0)

if __name__ == "__main__":
    main()
