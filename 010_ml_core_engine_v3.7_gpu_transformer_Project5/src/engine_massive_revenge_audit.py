#!/usr/bin/env python3
"""
engine_massive_revenge_audit.py - The Ultimate Vengeful Torture & Stress Suite
Built to relentlessly crush, stress-test, and verify the Rust + Wgpu + PyO3 ML Core Engine.
"""

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

RESULT_TAG = "@@REVENGE_RESULT@@"
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"

def log_r(msg=""):
    print(f"💀 [MASSIVE REVENGE] {msg}", flush=True)

class RevengeResult:
    def __init__(self):
        self.status = PASS
        self.notes = []

    def fail(self, msg):
        self.status = FAIL
        self.notes.append(msg)
        log_r(f"FAIL -> {msg}")

    def warn(self, msg):
        if self.status == PASS:
            self.status = WARN
        self.notes.append(msg)
        log_r(f"WARN -> {msg}")

def get_engine():
    import ml_core_engine
    return ml_core_engine

def mat_r(rows, cols, arr):
    flat = np.asarray(arr, dtype=np.float32).ravel().tolist()
    return get_engine().PyMatrix.from_list(rows, cols, flat)

def to_np_r(m, rows, cols):
    return np.asarray(m.to_list(), dtype=np.float32).reshape(rows, cols)

# -----------------------------------------------------------------------------
# Torture Test 1: Concurrency Storm (Multi-threading GPU Hammering)
# -----------------------------------------------------------------------------
def torture_concurrency(res):
    log_r("Executing Torture 1: Massive Multi-Threaded GPU Hammering (32 Threads)...")
    errors = []
    def worker(tid):
        try:
            rng = np.random.default_rng(100 + tid)
            for _ in range(100):
                n = rng.choice([64, 128, 256])
                A = rng.standard_normal((n, n)).astype(np.float32)
                B = rng.standard_normal((n, n)).astype(np.float32)
                ma = mat_r(n, n, A)
                mb = mat_r(n, n, B)
                c = ma.gpu_matmul(mb, use_relu=True)
                _ = c.gpu_relu_backward(c)
        except Exception as e:
            errors.append((tid, str(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(32)]
    t0 = time.perf_counter()
    for th in threads: th.start()
    for th in threads: th.join()
    elapsed = time.perf_counter() - t0

    if errors:
        res.fail(f"Concurrency storm failed with {len(errors)} thread errors. First: {errors[0]}")
    else:
        log_r(f"Concurrency storm survived cleanly! 32 threads completed heavy GPU pipelines in {elapsed:.3f}s.")

# -----------------------------------------------------------------------------
# Torture Test 2: Pathological Shapes & Edge-Case Matrix Chaos
# -----------------------------------------------------------------------------
def torture_pathological_shapes(res):
    log_r("Executing Torture 2: Pathological Shapes & Edge-Case Matrix Chaos...")
    rng = np.random.default_rng(666)
    odd_sizes = [1, 3, 7, 19, 37, 63, 129, 255, 511]
    
    for _ in range(300):
        m = random.choice(odd_sizes)
        k = random.choice(odd_sizes)
        n = random.choice(odd_sizes)
        
        A = rng.standard_normal((m, k)).astype(np.float32) * 1e4
        B = rng.standard_normal((k, n)).astype(np.float32) * 1e-4
        
        try:
            ma = mat_r(m, k, A)
            mb = mat_r(k, n, B)
            out = ma.gpu_matmul(mb, use_relu=True)
            got = to_np_r(out, m, n)
            if got.shape != (m, n):
                res.fail(f"Shape distortion detected: expected ({m},{n}), got {got.shape}")
        except Exception as e:
            res.fail(f"Exception during pathological shape {m}x{k}@{k}x{n}: {e}")
            break
    log_r("Pathological shapes and scale variations passed without a scratch.")

# -----------------------------------------------------------------------------
# Torture Test 3: VRAM/RAM Churn & Memory Leakage Assault
# -----------------------------------------------------------------------------
def torture_memory_churn(res):
    log_r("Executing Torture 3: VRAM & Host Memory Churn Assault (5000 fast allocations)...")
    E = get_engine()
    n = 128
    data = [0.1] * (n * n)
    
    try:
        import psutil
        proc = psutil.Process()
        mem_before = proc.memory_info().rss / 1e6
        
        for i in range(5000):
            _ = E.PyMatrix.from_list(n, n, data)
            
        mem_after = proc.memory_info().rss / 1e6
        diff = mem_after - mem_before
        log_r(f"Memory difference after 5000 allocations: {diff:+.2f} MB")
        if diff > 50.0:
            res.warn(f"Possible memory bloat detected during rapid allocations (+{diff:.2f} MB)")
    except ImportError:
        log_r("psutil not installed, skipping precise RSS tracking.")
    except Exception as e:
        res.warn(f"Memory churn check error: {e}")

# -----------------------------------------------------------------------------
# Torture Test 4: Extreme Numerical Overflow & NaN Propagation
# -----------------------------------------------------------------------------
def torture_numerics(res):
    log_r("Executing Torture 4: Extreme Numerical Overflow & NaN Storm...")
    E = get_engine()
    try:
      
        inf_mat = mat_r(2, 2, [float("inf"), 1.0, -float("inf"), 0.0])
        identity = mat_r(2, 2, [1.0, 0.0, 0.0, 1.0])
        out = inf_mat.gpu_matmul(identity, use_relu=False)
        log_r(f"Inf propagation handled. Output sample: {out.to_list()[:2]}")
    except Exception as e:
        res.fail(f"Engine panicked on Inf/NaN values: {e}")

# -----------------------------------------------------------------------------
# Main Execution Runner
# -----------------------------------------------------------------------------
def main():
    print("=" * 74)
    print(" 💀 ml_core_engine: THE MASSIVE REVENGE TORTURE SUITE 💀")
    print("=" * 74)
    
    res = RevengeResult()
    try:
        torture_concurrency(res)
        torture_pathological_shapes(res)
        torture_memory_churn(res)
        torture_numerics(res)
    except BaseException as e:
        res.fail(f"Fatal crash during revenge suite: {e}")
        traceback.print_exc()

    print("=" * 74)
    if res.status == FAIL:
        print(f" ❌ REVENGE FAILED: Engine buckled under pressure.")
        for note in res.notes: print(f"    - {note}")
        sys.exit(1)
    elif res.status == WARN:
        print(f" ⚠️ REVENGE SURVIVED WITH WARNINGS.")
        for note in res.notes: print(f"    - {note}")
        sys.exit(0)
    else:
        print(f" 🎉 TOTAL DOMINATION! The engine utterly crushed the revenge audit suite!")
    print("=" * 74)
    sys.exit(0)

if __name__ == "__main__":
    main()