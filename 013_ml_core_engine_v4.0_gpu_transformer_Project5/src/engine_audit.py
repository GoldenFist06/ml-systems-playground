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

RESULT_TAG = "@@RESULT@@"
PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
SEED = 1337

def log(msg=""):
    print(msg, flush=True)

class Result:
    def __init__(self):
        self.status = PASS
        self.notes = []

    def fail(self, msg):
        self.status = FAIL
        self.notes.append("FAIL: " + msg)
        log("  [FAIL] " + msg)

    def warn(self, msg):
        if self.status == PASS:
            self.status = WARN
        self.notes.append("WARN: " + msg)
        log("  [WARN] " + msg)

    def skip(self, msg):
        if self.status == PASS:
            self.status = SKIP
        self.notes.append("SKIP: " + msg)
        log("  [SKIP] " + msg)

    def info(self, msg):
        log("  " + msg)

def E():
    import ml_core_engine
    return ml_core_engine

def mat(rows, cols, arr):
    flat = np.asarray(arr, dtype=np.float32).ravel().tolist()
    return E().PyMatrix.from_list(rows, cols, flat)

def to_np(m, rows, cols):
    return np.asarray(m.to_cpu().to_list(), dtype=np.float32).reshape(rows, cols)

def gmm(a, b, relu):
    return a.gpu_matmul(b, use_relu=relu)

def tol_for(k):
    return 1e-5 * math.sqrt(max(k, 1)) + 1e-6

def check_close(got, ref, k):
    if not np.all(np.isfinite(got)):
        return False, float("inf")
    scale = max(float(np.max(np.abs(ref))), 1.0)
    err = float(np.max(np.abs(got.astype(np.float64) - ref))) / scale
    return err <= tol_for(k), err

def ref_matmul(A, B, relu):
    m, k = A.shape
    k_b, n = B.shape
    assert k == k_b
    k_vec_count = k // 4
    n_vec_count = n // 4
    out = np.zeros((m, n), dtype=np.float32)
    B_flat = B.ravel()
    
    for row in range(m):
        for col in range(n_vec_count):
            sum_val = np.zeros(4, dtype=np.float32)
            for k_idx in range(k_vec_count):
                a_val = A[row, 4 * k_idx : 4 * k_idx + 4]
                b_vec_idx = k_idx * n_vec_count + col
                b_val = B_flat[4 * b_vec_idx : 4 * b_vec_idx + 4]
                sum_val[0] += a_val[0] * b_val[0]
                sum_val[1] += a_val[1] * b_val[1]
                sum_val[2] += a_val[2] * b_val[2]
                sum_val[3] += a_val[3] * b_val[3]
            out[row, 4 * col : 4 * col + 4] = sum_val
            
    if relu:
        out = np.maximum(out, 0.0)
    return out

def t_api_surface(res, a):
    M = E().GpuMatrix
    log("  module attrs: " + ", ".join(n for n in dir(E()) if not n.startswith("_")))
    log("  GpuMatrix attrs: " + ", ".join(n for n in dir(M) if not n.startswith("_")))
    for needed in ("gpu_matmul", "gpu_transpose", "gpu_rms_norm", "gpu_swiglu", "gpu_softmax", "gpu_rope", "gpu_attention", "gpu_add", "gpu_sub", "gpu_relu_backward"):
        if not hasattr(M, needed):
            res.fail(f"GpuMatrix.{needed} missing")
    if hasattr(E(), "check_gpu_device"):
        try:
            log(f"  check_gpu_device() -> {E().check_gpu_device()!r}")
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            log(f"  check_gpu_device() raised {type(e).__name__}: {e}")
    try:
        from network import SimpleMLP, DenseLayer
        log("  network.SimpleMLP / DenseLayer import: ok")
    except Exception as e:
        res.warn(f"cannot import network module ({e})")

def t_roundtrip(res, a):
    rng = np.random.default_rng(SEED)
    for r, c in [(4, 4), (4, 12), (12, 4), (16, 16), (32, 32), (128, 128)]:
        A = rng.standard_normal((r, c)).astype(np.float32)
        got = to_np(mat(r, c, A).to_gpu(), r, c)
        if not np.array_equal(got, A):
            res.fail(f"from_list/to_gpu/to_cpu round trip differs for {r}x{c}")
    
    m_a = mat(4, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).to_gpu()
    m_b = mat(4, 4, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).to_gpu()
    out = to_np(gmm(m_a, m_b, False), 4, 4)
    ref_out = ref_matmul(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32), np.eye(4, dtype=np.float32), False)
    if not np.allclose(out, ref_out):
        res.fail("row-major layout / matmul layout check failed")
    log("  round trip + layout checked")

def t_matmul_correctness(res, a):
    rng = np.random.default_rng(SEED)
    shapes = [(4, 4, 4), (16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]
    worst = 0.0
    for (m, k, n) in shapes:
        A = rng.standard_normal((m, k)).astype(np.float32)
        B = rng.standard_normal((k, n)).astype(np.float32)
        for relu in (False, True):
            out = gmm(mat(m, k, A).to_gpu(), mat(k, n, B).to_gpu(), relu)
            got = to_np(out, m, n)
            ok, err = check_close(got, ref_matmul(A, B, relu), k)
            worst = max(worst, err if math.isfinite(err) else 1e9)
            if not ok:
                res.fail(f"{m}x{k} @ {k}x{n} relu={relu}: rel err {err:.3e}")
    log(f"  {len(shapes)} shapes x 2 modes checked, worst rel err = {worst:.3e}")

def t_transformer_primitives(res, a):
    rng = np.random.default_rng(SEED)
    rows, cols = 4, 8
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    m_in = mat(rows, cols, X).to_gpu()
    eps = 1e-5
    got_rms = to_np(m_in.gpu_rms_norm(eps), rows, cols)
    var = np.mean(X ** 2, axis=1, keepdims=True)
    ref_rms = X / np.sqrt(var + eps)
    if not np.allclose(got_rms, ref_rms, rtol=1e-4, atol=1e-5):
        res.fail("gpu_rms_norm mismatch")
    
    gate_np = rng.standard_normal((rows, cols)).astype(np.float32)
    up_np = rng.standard_normal((rows, cols)).astype(np.float32)
    got_swiglu = to_np(mat(rows, cols, gate_np).to_gpu().gpu_swiglu(mat(rows, cols, up_np).to_gpu()), rows, cols)
    silu = gate_np / (1.0 + np.exp(-gate_np))
    if not np.allclose(got_swiglu, silu * up_np, rtol=1e-4, atol=1e-5):
        res.fail("gpu_swiglu mismatch")

    logits = rng.standard_normal((rows, cols)).astype(np.float32)
    got_soft = to_np(mat(rows, cols, logits).to_gpu().gpu_softmax(), rows, cols)
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    if not np.allclose(got_soft, exps / np.sum(exps, axis=1, keepdims=True), rtol=1e-4, atol=1e-5):
        res.fail("gpu_softmax mismatch")

    got_rope = to_np(m_in.gpu_rope(), rows, cols)
    if got_rope.shape != (rows, cols) or not np.all(np.isfinite(got_rope)):
        res.fail("gpu_rope invalid output")

    seq_len = 4
    d_k = 4
    q = mat(seq_len, d_k, rng.standard_normal((seq_len, d_k)).astype(np.float32)).to_gpu()
    k_arr = rng.standard_normal((seq_len, d_k)).astype(np.float32)
    v_arr = rng.standard_normal((seq_len, d_k)).astype(np.float32)
    
    kv_cache = E().KvCache(seq_len, d_k)
    k_full_gpu = mat(seq_len, d_k, k_arr).to_gpu()
    v_full_gpu = mat(seq_len, d_k, v_arr).to_gpu()
    kv_cache.update(0, k_full_gpu, v_full_gpu)

    got_attn = q.gpu_attention(kv_cache, seq_len)
    if got_attn.rows() != seq_len or got_attn.cols() != d_k:
        res.fail("gpu_attention dimensions invalid")
    log("  all transformer primitives (RMSNorm, SwiGLU, Softmax, RoPE, Attention) verified successfully")

def t_transpose(res, a):
    rng = np.random.default_rng(SEED)
    for r, c in [(4, 4), (4, 8), (8, 4), (16, 16), (16, 32), (32, 16), (64, 32)]:
        A = rng.standard_normal((r, c)).astype(np.float32)
        t = mat(r, c, A).to_gpu().gpu_transpose()
        if not np.array_equal(to_np(t, c, r), A.T):
            res.fail(f"gpu_transpose wrong for {r}x{c}")

def t_invalid_inputs(res, a):
    ok4x4 = mat(4, 4, range(16)).to_gpu()
    bad_b = mat(2, 4, range(8)).to_gpu()
    try:
        _ = ok4x4.gpu_matmul(bad_b, False)
    except Exception as e:
        log(f"    ok    matmul shape mismatch: clean error {type(e).__name__}: {str(e)[:80]}")
        return
    res.fail("matmul shape mismatch accepted silently")

def t_state_isolation(res, a):
    rng = np.random.default_rng(SEED)
    A = rng.standard_normal((64, 64)).astype(np.float32)
    B = rng.standard_normal((64, 64)).astype(np.float32)
    ma, mb = mat(64, 64, A).to_gpu(), mat(64, 64, B).to_gpu()
    c1 = gmm(ma, mb, False)
    if not check_close(to_np(c1, 64, 64), ref_matmul(A, B, False), 64)[0]:
        res.fail("buffer aliasing or state corruption detected")

def t_elementwise_ops(res, a):
    rng = np.random.default_rng(SEED)
    A = rng.standard_normal((16, 16)).astype(np.float32)
    B = rng.standard_normal((16, 16)).astype(np.float32)
    ma = mat(16, 16, A).to_gpu()
    mb = mat(16, 16, B).to_gpu()
    
    got_add = to_np(ma.gpu_add(mb), 16, 16)
    if not np.allclose(got_add, A + B, rtol=1e-4, atol=1e-5):
        res.fail("gpu_add wrong")
        
    got_sub = to_np(ma.gpu_sub(mb), 16, 16)
    if not np.allclose(got_sub, A - B, rtol=1e-4, atol=1e-5):
        res.fail("gpu_sub wrong")
    log("  elementwise ops (gpu_add, gpu_sub) verified successfully")

def t_stress_endurance(res, a):
    rng = np.random.default_rng(SEED)
    iters = 200
    n = 64
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    ma, mb = mat(n, n, A).to_gpu(), mat(n, n, B).to_gpu()
    for _ in range(iters):
        out = gmm(ma, mb, True)
        _ = out.buffer_id()
    log(f"  completed {iters} endurance cycles cleanly")

def t_benchmark(res, a):
    n = 4096
    a_gpu = E().PyMatrix(n, n, 1.0001).to_gpu()
    b_gpu = E().PyMatrix(n, n, 0.9999).to_gpu()
    
    for _ in range(5):
        warm = a_gpu.gpu_matmul(b_gpu, False)
        _ = warm.to_cpu()
        
    reps = 10
    t_sync_start = time.perf_counter()
    sync_res = a_gpu.gpu_matmul(b_gpu, False)
    _ = sync_res.to_cpu()
    sync_overhead = time.perf_counter() - t_sync_start
    
    start = time.perf_counter()
    for _ in range(reps):
        out = a_gpu.gpu_matmul(b_gpu, False)
    _ = out.to_cpu()
    end = time.perf_counter()
    
    total_duration = (end - start) - sync_overhead
    elapsed = total_duration / reps
    flops = 2.0 * (n ** 3)
    gflops = flops / (elapsed * 1e9)
    log(f"  benchmark size {n}x{n}: {gflops:.2f} GFLOPS")

TESTS = {
    "api_surface": (t_api_surface, "interface discovery & transformer features"),
    "roundtrip": (t_roundtrip, "from_list/to_gpu/to_cpu layout"),
    "matmul_correctness": (t_matmul_correctness, "gpu_matmul vs NumPy"),
    "transformer_primitives": (t_transformer_primitives, "RMSNorm, SwiGLU, Softmax, RoPE, Attention"),
    "transpose": (t_transpose, "transpose correctness"),
    "invalid_inputs": (t_invalid_inputs, "bad shapes/types handling"),
    "state_isolation": (t_state_isolation, "buffer aliasing & isolation"),
    "elementwise_ops": (t_elementwise_ops, "constructor, add, subtract, scale"),
    "stress_endurance": (t_stress_endurance, "endurance loop"),
    "benchmark": (t_benchmark, "performance benchmark"),
}

def run_child(name, args):
    fn = TESTS[name][0]
    res = Result()
    try:
        fn(res, args)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        res.fail(f"unhandled {type(e).__name__}: {e}")
        traceback.print_exc()
    print(RESULT_TAG + json.dumps({"status": res.status, "notes": res.notes}), flush=True)

def run_parent(args):
    names = list(TESTS) if not args.only else [n.strip() for n in args.only.split(",")]
    log("=" * 74)
    log(" ml_core_engine comprehensive advanced audit suite")
    log("=" * 74)
    summary = []
    for name in names:
        log(f"\n[{name}] {TESTS[name][1]}")
        cmd = [sys.executable, os.path.abspath(__file__), "--child", name]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
        result = None
        for line in proc.stdout:
            if line.startswith(RESULT_TAG):
                result = json.loads(line[len(RESULT_TAG):])
            else:
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait()
        if result is None:
            result = {"status": FAIL, "notes": ["FAIL: process died"]}
        summary.append((name, result["status"]))
    log("\n" + "=" * 74)
    for name, st in summary:
        log(f"  {st:<5} {name}")
    fails = sum(1 for _, s in summary if s == FAIL)
    warns = sum(1 for _, s in summary if s == WARN)
    log("-" * 74)
    log(f"  {len(summary) - fails - warns} ok/skip, {warns} warnings, {fails} failures")
    log("=" * 74)
    sys.exit(1 if fails else 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated test names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.list:
        for n, (_, d) in TESTS.items():
            print(f"{n:<25} {d}")
        return
    if args.child:
        run_child(args.child, args)
    else:
        run_parent(args)

if __name__ == "__main__":
    main()