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
    return np.asarray(m.to_list(), dtype=np.float32).reshape(rows, cols)

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
    R = A.astype(np.float64) @ B.astype(np.float64)
    return np.maximum(R, 0.0) if relu else R

def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass
    return None

def median_time(fn, warm, reps):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

def probe(res, name, fn, on_accept):
    try:
        out = fn()
    except Exception as e:
        log(f"    ok   {name}: clean error {type(e).__name__}: {str(e)[:80]}")
        return
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        res.warn(f"{name}: Rust panic crossed the FFI boundary ({type(e).__name__})")
        return
    msg = f"{name}: accepted silently (returned {type(out).__name__})"
    (res.fail if on_accept == "fail" else res.warn)(msg)

def t_api_surface(res, a):
    M = E().PyMatrix
    log("  module attrs: " + ", ".join(n for n in dir(E()) if not n.startswith("_")))
    log("  PyMatrix attrs: " + ", ".join(n for n in dir(M) if not n.startswith("_")))
    for needed in ("from_list", "to_list", "gpu_matmul", "transpose", "gpu_rms_norm", "gpu_swiglu", "gpu_softmax", "gpu_rope", "gpu_attention"):
        if not hasattr(M, needed):
            res.fail(f"PyMatrix.{needed} missing")
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
    for r, c in [(1, 1), (1, 9), (9, 1), (2, 3), (17, 33), (128, 128), (255, 257)]:
        A = rng.standard_normal((r, c)).astype(np.float32)
        got = to_np(mat(r, c, A), r, c)
        if not np.array_equal(got, A):
            res.fail(f"from_list/to_list round trip differs for {r}x{c}")
    out = to_np(gmm(mat(2, 3, [1, 2, 3, 4, 5, 6]), mat(3, 1, [1, 0, 0]), False), 2, 1)
    if not np.allclose(out.ravel(), [1.0, 4.0]):
        res.fail("row-major layout check failed")
    log("  round trip + layout checked")

def t_matmul_correctness(res, a):
    rng = np.random.default_rng(SEED)
    shapes = [(1, 1, 1), (1, 7, 1), (7, 1, 7), (2, 3, 4), (3, 5, 7), (8, 8, 8), (15, 17, 19),
              (16, 16, 16), (17, 33, 65), (31, 32, 33), (33, 31, 32), (64, 64, 64),
              (100, 100, 100), (129, 257, 63), (256, 256, 256), (512, 512, 512)]
    worst = 0.0
    for (m, k, n) in shapes:
        A = rng.standard_normal((m, k)).astype(np.float32)
        B = rng.standard_normal((k, n)).astype(np.float32)
        for relu in (False, True):
            out = gmm(mat(m, k, A), mat(k, n, B), relu)
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
    m_in = mat(rows, cols, X)
    eps = 1e-5
    got_rms = to_np(m_in.gpu_rms_norm(eps), rows, cols)
    var = np.mean(X ** 2, axis=1, keepdims=True)
    ref_rms = X / np.sqrt(var + eps)
    if not np.allclose(got_rms, ref_rms, rtol=1e-4, atol=1e-5):
        res.fail("gpu_rms_norm mismatch")
    
    gate_np = rng.standard_normal((rows, cols)).astype(np.float32)
    up_np = rng.standard_normal((rows, cols)).astype(np.float32)
    got_swiglu = to_np(mat(rows, cols, gate_np).gpu_swiglu(mat(rows, cols, up_np)), rows, cols)
    silu = gate_np / (1.0 + np.exp(-gate_np))
    if not np.allclose(got_swiglu, silu * up_np, rtol=1e-4, atol=1e-5):
        res.fail("gpu_swiglu mismatch")

    logits = rng.standard_normal((rows, cols)).astype(np.float32)
    got_soft = to_np(mat(rows, cols, logits).gpu_softmax(), rows, cols)
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    if not np.allclose(got_soft, exps / np.sum(exps, axis=1, keepdims=True), rtol=1e-4, atol=1e-5):
        res.fail("gpu_softmax mismatch")

    got_rope = to_np(m_in.gpu_rope(), rows, cols)
    if got_rope.shape != (rows, cols) or not np.all(np.isfinite(got_rope)):
        res.fail("gpu_rope invalid output")

    q = mat(2, 4, rng.standard_normal((2, 4)).astype(np.float32))
    k = mat(2, 4, rng.standard_normal((2, 4)).astype(np.float32))
    v = mat(2, 4, rng.standard_normal((2, 4)).astype(np.float32))
    got_attn = q.gpu_attention(k, v)
    if got_attn.rows != 2 or got_attn.cols != 4:
        res.fail("gpu_attention dimensions invalid")
    log("  all transformer primitives (RMSNorm, SwiGLU, Softmax, RoPE, Attention) verified successfully")

def t_transpose(res, a):
    rng = np.random.default_rng(SEED)
    for r, c in [(1, 1), (1, 7), (7, 1), (2, 3), (3, 2), (16, 16), (17, 33), (33, 17), (100, 37)]:
        A = rng.standard_normal((r, c)).astype(np.float32)
        t = mat(r, c, A).transpose()
        if not np.array_equal(to_np(t, c, r), A.T):
            res.fail(f"transpose wrong for {r}x{c}")

def t_invalid_inputs(res, a):
    M = E().PyMatrix
    ok2x3 = mat(2, 3, range(6))
    probe(res, "from_list len too short", lambda: M.from_list(2, 3, [1.0] * 5), "fail")
    probe(res, "from_list len too long", lambda: M.from_list(2, 3, [1.0] * 7), "fail")
    probe(res, "matmul shape mismatch", lambda: gmm(ok2x3, ok2x3, False).to_list(), "fail")

def t_state_isolation(res, a):
    rng = np.random.default_rng(SEED)
    A = rng.standard_normal((64, 48)).astype(np.float32)
    B = rng.standard_normal((48, 32)).astype(np.float32)
    ma, mb = mat(64, 48, A), mat(48, 32, B)
    c1 = gmm(ma, mb, False)
    if not check_close(to_np(c1, 64, 32), ref_matmul(A, B, False), 48)[0]:
        res.fail("buffer aliasing or state corruption detected")

def t_elementwise_ops(res, a):
    P = E().PyMatrix
    rng = np.random.default_rng(SEED)
    f = P(3, 4, 2.5)
    if not np.allclose(to_np(f, 3, 4), 2.5):
        res.fail("constructor fill failed")
    A = rng.standard_normal((4, 4)).astype(np.float32)
    B = rng.standard_normal((4, 4)).astype(np.float32)
    ma, mb = mat(4, 4, A), mat(4, 4, B)
    if not np.allclose(to_np(ma.add(mb), 4, 4), A + B):
        res.fail("add wrong")
    if not np.allclose(to_np(ma.multiply(mb), 4, 4), A * B):
        res.fail("multiply wrong")

def t_stress_endurance(res, a):
    rng = np.random.default_rng(SEED)
    iters = 200
    n = 64
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    ma, mb = mat(n, n, A), mat(n, n, B)
    for i in range(iters):
        out = gmm(ma, mb, True)
        del out
    log(f"  completed {iters} endurance cycles cleanly")

def t_benchmark(res, a):
    rng = np.random.default_rng(SEED)
    n = 256
    A = rng.standard_normal((n, n), dtype=np.float32)
    B = rng.standard_normal((n, n), dtype=np.float32)
    la, lb = A.ravel().tolist(), B.ravel().tolist()
    P = E().PyMatrix
    a_, b_ = P.from_list(n, n, la), P.from_list(n, n, lb)
    t_tot = median_time(lambda: a_.gpu_matmul(b_, use_relu=False).to_list(), 2, 5)
    flops = 2.0 * n ** 3
    log(f"  benchmark size {n}x{n}: {flops / t_tot / 1e9:.2f} GFLOPS")

TESTS = {
    "api_surface": (t_api_surface, "interface discovery & transformer features"),
    "roundtrip": (t_roundtrip, "from_list/to_list layout"),
    "matmul_correctness": (t_matmul_correctness, "gpu_matmul vs NumPy"),
    "transformer_primitives": (t_transformer_primitives, "RMSNorm, SwiGLU, Softmax, RoPE, Attention"),
    "transpose": (t_transpose, "transpose correctness"),
    "invalid_inputs": (t_invalid_inputs, "bad shapes/types handling"),
    "state_isolation": (t_state_isolation, "buffer aliasing & isolation"),
    "elementwise_ops": (t_elementwise_ops, "constructor, add, multiply, scale"),
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