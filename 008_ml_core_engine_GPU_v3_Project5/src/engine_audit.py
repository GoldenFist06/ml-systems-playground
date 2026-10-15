#!/usr/bin/env python3
"""
engine_audit.py - independent black-box audit for ml_core_engine (Rust + wgpu + PyO3).

Put this file next to network.py (project folder), build the wheel in release mode:

    maturin develop --release

then run:

    python engine_audit.py                 # full suite
    python engine_audit.py --quick         # smaller sizes / fewer iterations
    python engine_audit.py --only matmul_correctness,benchmark
    python engine_audit.py --big           # adds 2048 benchmark (needs several GB of RAM)
    python engine_audit.py --list

Design notes
  * Every test runs in its own child process. A hard crash in Rust/wgpu (abort, access
    violation, device lost) therefore only fails that one test instead of killing the run.
  * A Rust panic surfaces in Python as pyo3_runtime.PanicException, which derives from
    BaseException (NOT Exception). We catch BaseException so it is reported, not fatal.
  * Everything is float32 (WGSL has no f64 in wgpu), so comparisons use a relative
    tolerance against a float64 NumPy reference.
  * Interface used (from the example script and network.py): PyMatrix(rows, cols, fill),
    PyMatrix.from_list(rows, cols, flat_list), .to_list(), .gpu_matmul(other, use_relu=False),
    .transpose(), .multiply(other), .scale(x), .sub(other), .rows, .cols, plus network.DenseLayer /
    SimpleMLP (layer.weights, layer.output, layer.backward(dvalues, learning_rate)).
    Row-major layout is assumed and verified by the tests.
  * Loss convention follows network.py: gradient = 2*(p - t) (sum of squares, NOT averaged by batch size).
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

RESULT_TAG = "@@RESULT@@"
PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
SEED = 1337


# ----------------------------------------------------------------------------- helpers
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
    # float32 accumulation error grows roughly with sqrt(k); error is measured relative
    # to max(|reference|, 1).
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
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
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
    """Run fn(); classify: clean Python exception = good, Rust panic = warn,
    silently accepted = on_accept ('warn' or 'fail')."""
    try:
        out = fn()
    except Exception as e:
        log(f"    ok   {name}: clean error {type(e).__name__}: {str(e)[:80]}")
        return
    except BaseException as e:  # PanicException lands here
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        res.warn(f"{name}: Rust panic crossed the FFI boundary ({type(e).__name__}); "
                 f"return a PyErr instead of panicking")
        return
    msg = f"{name}: accepted silently (returned {type(out).__name__})"
    (res.fail if on_accept == "fail" else res.warn)(msg)


# ----------------------------------------------------------------------------- tests
def t_api_surface(res, a):
    M = E().PyMatrix
    log("  module attrs: " + ", ".join(n for n in dir(E()) if not n.startswith("_")))
    log("  PyMatrix attrs: " + ", ".join(n for n in dir(M) if not n.startswith("_")))
    for needed in ("from_list", "to_list", "gpu_matmul", "transpose"):
        if not hasattr(M, needed):
            res.fail(f"PyMatrix.{needed} missing")
    if hasattr(E(), "check_gpu_device"):
        try:
            log(f"  check_gpu_device() -> {E().check_gpu_device()!r}")
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            log(f"  check_gpu_device() raised {type(e).__name__}: {e}")
    m = mat(2, 3, range(6))
    for attr, want in (("rows", 2), ("cols", 3)):
        if not hasattr(m, attr):
            res.warn(f"PyMatrix.{attr} not exposed")
        elif getattr(m, attr) != want:
            res.fail(f"PyMatrix.{attr} = {getattr(m, attr)}, expected {want}")
    try:
        from network import SimpleMLP, DenseLayer  # noqa: F401
        log("  network.SimpleMLP / DenseLayer import: ok")
    except Exception as e:
        res.warn(f"cannot import network module ({e}); MLP tests will fail")


def t_roundtrip(res, a):
    rng = np.random.default_rng(SEED)
    for r, c in [(1, 1), (1, 9), (9, 1), (2, 3), (17, 33), (128, 128), (255, 257)]:
        A = rng.standard_normal((r, c)).astype(np.float32)
        got = to_np(mat(r, c, A), r, c)
        if not np.array_equal(got, A):
            res.fail(f"from_list/to_list round trip differs for {r}x{c}")
    # layout: [[1,2,3],[4,5,6]] @ [1,0,0]^T must give [1,4] if layout is row-major
    out = to_np(gmm(mat(2, 3, [1, 2, 3, 4, 5, 6]), mat(3, 1, [1, 0, 0]), False), 2, 1)
    if not np.allclose(out.ravel(), [1.0, 4.0]):
        res.fail(f"row-major layout check failed: got {out.ravel().tolist()}, expected [1, 4]")
    log("  round trip + layout checked")


def t_matmul_correctness(res, a):
    rng = np.random.default_rng(SEED)
    shapes = [(1, 1, 1), (1, 7, 1), (7, 1, 7), (2, 3, 4), (3, 5, 7), (8, 8, 8), (15, 17, 19),
              (16, 16, 16), (17, 33, 65), (31, 32, 33), (33, 31, 32), (64, 64, 64),
              (100, 100, 100), (129, 257, 63), (256, 256, 256), (512, 512, 512)]
    if not a.quick:
        shapes.append((1024, 1024, 1024))
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
                res.fail(f"{m}x{k} @ {k}x{n} relu={relu}: rel err {err:.3e} > {tol_for(k):.1e}")
    log(f"  {len(shapes)} shapes x 2 modes checked, worst rel err = {worst:.3e}")


def t_relu_extremes(res, a):
    vals = np.array([-1e5, -1.0, -1e-7, 0.0, 1e-7, 1.0, 1e5], dtype=np.float32)
    A = np.stack([np.roll(vals, i) for i in range(7)])          # 7x7, every column sees every value
    I = np.eye(7, dtype=np.float32)
    out = to_np(gmm(mat(7, 7, A), mat(7, 7, I), True), 7, 7)
    if np.any(out < 0):
        res.fail("ReLU output contains negative values")
    if not np.allclose(out, np.maximum(A, 0.0), rtol=1e-6, atol=0.0):
        res.fail("ReLU(A @ I) != max(A, 0) on extreme values")
    out2 = to_np(gmm(mat(7, 7, A), mat(7, 7, I), False), 7, 7)
    if not np.allclose(out2, A, rtol=1e-6, atol=0.0):
        res.fail("A @ I != A without ReLU (negatives must survive)")
    rng = np.random.default_rng(SEED)
    N = -np.abs(rng.standard_normal((20, 20))).astype(np.float32)
    P = np.abs(rng.standard_normal((20, 20))).astype(np.float32)
    ones = np.ones((20, 20), dtype=np.float32)
    # N @ P with all-negative A and all-positive B -> all outputs <= 0 -> ReLU must give zeros
    z = to_np(gmm(mat(20, 20, N), mat(20, 20, P), True), 20, 20)
    if np.any(z != 0):
        res.fail("all-negative pre-activation did not produce exact zeros")
    pos = to_np(gmm(mat(20, 20, P), mat(20, 20, ones), True), 20, 20)
    if not check_close(pos, ref_matmul(P, ones, True), 20)[0]:
        res.fail("all-positive pre-activation altered by ReLU")
    log("  extreme values / sign handling checked")


def t_transpose(res, a):
    rng = np.random.default_rng(SEED)
    for r, c in [(1, 1), (1, 7), (7, 1), (2, 3), (3, 2), (16, 16), (17, 33), (33, 17),
                 (100, 37), (256, 129), (1024, 3)]:
        A = rng.standard_normal((r, c)).astype(np.float32)
        t = mat(r, c, A).transpose()
        got = to_np(t, c, r)
        if not np.array_equal(got, A.T):
            res.fail(f"transpose wrong for {r}x{c}")
            continue
        if hasattr(t, "rows") and (t.rows, t.cols) != (c, r):
            res.fail(f"transpose dims {(t.rows, t.cols)} != {(c, r)} for {r}x{c}")
        back = to_np(t.transpose(), r, c)
        if not np.array_equal(back, A):
            res.fail(f"double transpose != identity for {r}x{c}")
    log("  transpose values, dims and involution checked")


def t_special_values(res, a):
    nan, inf = float("nan"), float("inf")
    I2 = mat(2, 2, [1, 0, 0, 1])
    out = to_np(gmm(mat(2, 2, [nan, 1, 2, 3]), I2, False), 2, 2)
    if not np.isnan(out[0, 0]):
        res.warn("NaN input did not propagate to output (silent NaN loss)")
    out = to_np(gmm(mat(1, 2, [inf, 1.0]), mat(2, 1, [1, 1]), False), 1, 1)
    if not (np.isinf(out[0, 0]) and out[0, 0] > 0):
        res.warn(f"inf + 1 expected +inf, got {out[0, 0]}")
    out = to_np(gmm(mat(1, 2, [inf, -inf]), mat(2, 1, [1, 1]), False), 1, 1)
    if not np.isnan(out[0, 0]):
        res.warn(f"inf - inf expected NaN, got {out[0, 0]}")
    out = to_np(gmm(mat(1, 1, [1e30]), mat(1, 1, [1e30]), False), 1, 1)
    if not np.isinf(out[0, 0]):
        res.warn(f"f32 overflow expected inf, got {out[0, 0]}")
    out = to_np(gmm(mat(1, 1, [nan]), mat(1, 1, [1.0]), True), 1, 1)
    log(f"  relu(NaN) -> {out[0, 0]} (informational: WGSL max() with NaN is implementation-defined)")


def t_invalid_inputs(res, a):
    M = E().PyMatrix
    ok2x3 = mat(2, 3, range(6))
    probe(res, "from_list len too short", lambda: M.from_list(2, 3, [1.0] * 5), "fail")
    probe(res, "from_list len too long", lambda: M.from_list(2, 3, [1.0] * 7), "fail")
    probe(res, "from_list negative rows", lambda: M.from_list(-1, 3, []), "fail")
    probe(res, "from_list non-numeric", lambda: M.from_list(1, 2, ["a", "b"]), "fail")
    probe(res, "from_list None", lambda: M.from_list(1, 1, None), "fail")
    probe(res, "from_list nested list", lambda: M.from_list(1, 2, [[1.0], [2.0]]), "fail")
    probe(res, "from_list absurd dims", lambda: M.from_list(10 ** 9, 10 ** 9, [1.0]), "fail")
    probe(res, "from_list 0x0", lambda: M.from_list(0, 0, []), "warn")
    probe(res, "from_list 0x3", lambda: M.from_list(0, 3, []), "warn")
    probe(res, "from_list 3x0", lambda: M.from_list(3, 0, []), "warn")
    probe(res, "matmul shape mismatch (2x3 @ 2x3)", lambda: gmm(ok2x3, ok2x3, False).to_list(), "fail")
    probe(res, "matmul shape mismatch (1x7 @ 1x7)",
          lambda: gmm(mat(1, 7, range(7)), mat(1, 7, range(7)), True).to_list(), "fail")
    probe(res, "multiply shape mismatch", lambda: ok2x3.multiply(mat(3, 2, range(6))).to_list(), "fail")
    probe(res, "sub shape mismatch", lambda: ok2x3.sub(mat(3, 2, range(6))).to_list(), "fail")
    probe(res, "matmul other=int", lambda: ok2x3.gpu_matmul(5, use_relu=False), "fail")
    probe(res, "matmul other=None", lambda: ok2x3.gpu_matmul(None, use_relu=False), "fail")
    probe(res, "matmul use_relu='yes'", lambda: ok2x3.gpu_matmul(mat(3, 1, [1, 1, 1]), use_relu="yes"), "warn")
    try:
        z1, z2 = M.from_list(0, 3, []), M.from_list(3, 2, [1.0] * 6)
        probe(res, "matmul 0x3 @ 3x2", lambda: z1.gpu_matmul(z2, use_relu=False).to_list(), "warn")
    except Exception:
        pass
    # the engine must still work after all that abuse
    A = np.eye(3, dtype=np.float32)
    if not np.allclose(to_np(gmm(mat(3, 3, A), mat(3, 3, A), False), 3, 3), A):
        res.fail("engine returns wrong results after invalid-input probes (state corrupted)")
    else:
        log("  engine still healthy after invalid-input probes")


def t_state_isolation(res, a):
    rng = np.random.default_rng(SEED)
    A = rng.standard_normal((64, 48)).astype(np.float32)
    B = rng.standard_normal((48, 32)).astype(np.float32)
    D = rng.standard_normal((48, 32)).astype(np.float32)
    W = rng.standard_normal((32, 16)).astype(np.float32)
    ma, mb, md, mw = mat(64, 48, A), mat(48, 32, B), mat(48, 32, D), mat(32, 16, W)
    c1 = gmm(ma, mb, False)
    c2 = gmm(ma, md, True)            # computed AFTER c1; must not clobber c1
    if not check_close(to_np(c1, 64, 32), ref_matmul(A, B, False), 48)[0]:
        res.fail("earlier result changed after a later op (result buffer reuse / aliasing bug)")
    if not check_close(to_np(c2, 64, 32), ref_matmul(A, D, True), 48)[0]:
        res.fail("later result wrong")
    if not np.array_equal(to_np(c1, 64, 32), to_np(c1, 64, 32)):
        res.fail("reading the same matrix twice gives different data")
    for m_, arr, shp, nm in ((ma, A, (64, 48), "A"), (mb, B, (48, 32), "B"), (md, D, (48, 32), "D")):
        if not np.array_equal(to_np(m_, *shp), arr):
            res.fail(f"input matrix {nm} was mutated by gpu_matmul")
    chained = gmm(c1, mw, True)
    ref = ref_matmul(ref_matmul(A, B, False), W, True)
    if not check_close(to_np(chained, 64, 16), ref, 32)[0]:
        res.fail("chained op (result used as input without readback) is wrong")
    S = rng.standard_normal((32, 32)).astype(np.float32)
    ms = mat(32, 32, S)
    if not check_close(to_np(gmm(ms, ms, False), 32, 32), ref_matmul(S, S, False), 32)[0]:
        res.fail("self-multiplication (a @ a, same buffer as both inputs) is wrong")
    outs = [to_np(gmm(ma, mb, False), 64, 32) for _ in range(25)]
    if not all(np.array_equal(outs[0], o) for o in outs[1:]):
        res.warn("25 identical matmuls are not bit-identical (non-deterministic reduction?)")
    log("  aliasing, mutation, chaining and determinism checked")


def t_fuzz_shapes(res, a):
    rnd = random.Random(SEED)
    rng = np.random.default_rng(SEED)
    special = [1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
    count = 60 if a.quick else 200
    bad = 0
    for i in range(count):
        m, k, n = (rnd.choice(special) if rnd.random() < 0.6 else rnd.randint(1, 160)
                   for _ in range(3))
        relu = rnd.random() < 0.5
        scale = 10 ** rnd.uniform(-3, 3)
        A = (rng.standard_normal((m, k)) * scale).astype(np.float32)
        B = (rng.standard_normal((k, n)) * scale).astype(np.float32)
        got = to_np(gmm(mat(m, k, A), mat(k, n, B), relu), m, n)
        ok, err = check_close(got, ref_matmul(A, B, relu), k)
        if not ok:
            bad += 1
            if bad <= 10:
                res.fail(f"shape {m}x{k}@{k}x{n} relu={relu} scale={scale:.1e}: err {err:.2e}")
    log(f"  {count} random shapes/scales checked, {bad} mismatches")
    if bad > 10:
        res.fail(f"{bad} total mismatches (first 10 listed)")


def t_elementwise_ops(res, a):
    P = E().PyMatrix
    rng = np.random.default_rng(SEED)
    try:
        f = P(3, 4, 2.5)
        if not np.allclose(to_np(f, 3, 4), 2.5):
            res.fail("PyMatrix(rows, cols, fill) does not fill every element with 'fill'")
        if (f.rows, f.cols) != (3, 4):
            res.fail("PyMatrix(rows, cols, fill) reports wrong dims")
        if np.any(to_np(P(1, 5, 0.0), 1, 5) != 0):
            res.fail("PyMatrix(1, n, 0.0) is not all zeros")
    except Exception as e:
        res.fail(f"PyMatrix(rows, cols, fill) constructor failed: {type(e).__name__}: {e}")
    for (r, c) in [(1, 1), (1, 9), (9, 1), (4, 4), (17, 33), (100, 257), (512, 512)]:
        A = rng.standard_normal((r, c)).astype(np.float32)
        B = rng.standard_normal((r, c)).astype(np.float32)
        ma, mb = mat(r, c, A), mat(r, c, B)
        for name, got_m, ref in (("multiply", ma.multiply(mb), A * B),
                                 ("sub", ma.sub(mb), A - B),
                                 ("scale(0.37)", ma.scale(0.37), A * np.float32(0.37))):
            if not np.allclose(to_np(got_m, r, c), ref, rtol=1e-5, atol=1e-6):
                res.fail(f"{name} wrong for {r}x{c}")
        if not (np.array_equal(to_np(ma, r, c), A) and np.array_equal(to_np(mb, r, c), B)):
            res.fail(f"multiply/sub/scale mutated their operands for {r}x{c}")
    A = rng.standard_normal((8, 8)).astype(np.float32)
    ma = mat(8, 8, A)
    for k in (0.0, -1.5, 1e-30, 1e30):
        if not np.allclose(to_np(ma.scale(k), 8, 8), A * np.float32(k), rtol=1e-5, atol=0.0, equal_nan=True):
            res.fail(f"scale({k}) wrong")
    # gpu_matmul default must be PLAIN matmul: network.py uses it for dW = x^T @ dz without use_relu
    X = rng.standard_normal((16, 12)).astype(np.float32)
    Y = rng.standard_normal((12, 10)).astype(np.float32)
    try:
        d = to_np(mat(16, 12, X).gpu_matmul(mat(12, 10, Y)), 16, 10)
        if not check_close(d, ref_matmul(X, Y, False), 12)[0]:
            res.fail("gpu_matmul(other) with default use_relu is NOT a plain matmul "
                     "(weight gradients in DenseLayer.backward would be silently clipped)")
    except TypeError as e:
        res.fail(f"gpu_matmul(other) without use_relu not accepted: {e}")
    log("  constructor, multiply, sub, scale, operand immutability, default matmul semantics checked")


def _set_w(layer, W):
    layer.weights = mat(W.shape[0], W.shape[1], W)


def _get_w(layer):
    return to_np(layer.weights, layer.weights.rows, layer.weights.cols)


def t_layer_backward(res, a):
    from network import DenseLayer
    rng = np.random.default_rng(SEED)
    # ---- (1) finite-difference gradient check through the whole network ----------------
    sizes, N = [3, 8, 6, 1], 8
    random.seed(SEED)
    mlp = _build_mlp(sizes)
    if not hasattr(mlp.layers[0], "weights"):
        res.skip("layers expose no .weights; cannot run gradient check")
        return
    Ws = [(rng.standard_normal((sizes[i], sizes[i + 1])) * 0.7).astype(np.float32) for i in range(3)]
    Ws[-1] = np.abs(Ws[-1]) + 0.1                                  # keep the output ReLU alive
    for l, W in zip(mlp.layers, Ws):
        _set_w(l, W)
    X = rng.standard_normal((N, 3)).astype(np.float32)
    y = rng.uniform(0.5, 3.0, (N, 1)).astype(np.float32)
    Xm = mat(N, 3, X)

    def loss():
        p = to_np(mlp.forward(Xm), N, 1).astype(np.float64)
        return float(np.sum((p - y) ** 2))

    p0 = to_np(mlp.forward(Xm), N, 1)
    active = [float(np.mean(to_np(l.output, l.output.rows, l.output.cols) > 0)) for l in mlp.layers]
    log("  fraction of active ReLU units per layer: " + ", ".join(f"{x:.2f}" for x in active))
    if active[-1] == 0.0:
        res.skip("output ReLU dead for this init; gradient check uninformative")
        return
    eps, lr = 1e-2, 2e-4
    sample = {li: [(rng.integers(Ws[li].shape[0]), rng.integers(Ws[li].shape[1])) for _ in range(6)]
              for li in range(3)}
    numeric = {}
    for li, layer in enumerate(mlp.layers):
        for (i, j) in sample[li]:
            vals = []
            for sgn in (+1, -1):
                W = Ws[li].copy()
                W[i, j] += sgn * eps
                _set_w(layer, W)
                vals.append(loss())
            _set_w(layer, Ws[li])
            numeric[(li, i, j)] = (vals[0] - vals[1]) / (2 * eps)
    p = to_np(mlp.forward(Xm), N, 1)
    dv = mat(N, 1, 2.0 * (p - y))
    for layer in reversed(mlp.layers):
        dv = layer.backward(dv, learning_rate=lr)
    for li, layer in enumerate(mlp.layers):
        analytic = (Ws[li] - _get_w(layer)) / lr                   # update = W - lr * dW
        worst = 0.0
        for (i, j) in sample[li]:
            n_, a_ = numeric[(li, i, j)], float(analytic[i, j])
            worst = max(worst, abs(a_ - n_) / (max(abs(a_), abs(n_)) + 1e-2))
        log(f"  layer {li}: worst relative gradient error vs finite differences = {worst:.3e}")
        if worst > 3e-2:
            res.fail(f"layer {li}: analytic gradient disagrees with numeric gradient (err {worst:.2e})")

    # ---- (2) single-layer: weight update + gradient passed to the previous layer ---------
    lr2 = 0.5
    random.seed(SEED)
    L = DenseLayer(5, 4)
    xin = rng.standard_normal((6, 5)).astype(np.float32)
    L.forward(mat(6, 5, xin))
    out_np = to_np(L.output, 6, 4)
    dvn = rng.standard_normal((6, 4)).astype(np.float32)
    W_old = _get_w(L)
    dz = dvn * (out_np > 0)
    ret = to_np(L.backward(mat(6, 4, dvn), learning_rate=lr2), 6, 5)
    W_new = _get_w(L)
    if not np.allclose(W_new, W_old - lr2 * (xin.T @ dz), rtol=1e-4, atol=1e-5):
        res.fail("weight update != W - lr * x^T @ (dvalues * relu'(out))")
    exp_true, exp_stale = dz @ W_old.T, dz @ W_new.T
    e_true = np.max(np.abs(ret - exp_true)) / max(np.max(np.abs(exp_true)), 1e-6)
    e_stale = np.max(np.abs(ret - exp_stale)) / max(np.max(np.abs(exp_stale)), 1e-6)
    log(f"  returned gradient vs OLD weights: err {e_true:.2e} | vs UPDATED weights: err {e_stale:.2e}")
    if e_true > 1e-3:
        if e_stale <= 1e-3:
            res.fail("backward() returns dL/dx computed with the ALREADY-UPDATED weights; "
                     "must use the weights from the forward pass (compute dx before updating self.weights)")
        else:
            res.fail("gradient returned to the previous layer matches neither old nor new weights")

    # ---- (3) bias: exists but is it used? -------------------------------------------------
    if hasattr(L, "bias"):
        before = to_np(L.forward(mat(6, 5, xin)), 6, 4)
        L.bias = E().PyMatrix(1, 4, 100.0)
        after = to_np(L.forward(mat(6, 5, xin)), 6, 4)
        if np.array_equal(before, after):
            res.warn("layer.bias exists but has no effect on forward/backward (dead parameter; "
                     "the model has no bias terms)")


def t_cpu_gpu_parity(res, a):
    M = E().PyMatrix
    rng = np.random.default_rng(SEED)
    if hasattr(M, "matmul"):
        for (m, k, n) in [(1, 1, 1), (3, 5, 7), (17, 33, 65), (64, 64, 64), (129, 257, 63)]:
            A = rng.standard_normal((m, k)).astype(np.float32)
            B = rng.standard_normal((k, n)).astype(np.float32)
            try:
                cpu = to_np(mat(m, k, A).matmul(mat(k, n, B)), m, n)
            except TypeError as e:
                res.warn(f"matmul(other) signature differs from assumption: {e}")
                break
            if not check_close(cpu, ref_matmul(A, B, False), k)[0]:
                res.fail(f"CPU matmul wrong for {m}x{k}@{k}x{n}")
            gpu = to_np(gmm(mat(m, k, A), mat(k, n, B), False), m, n)
            if not check_close(gpu, cpu.astype(np.float64), k)[0]:
                res.fail(f"CPU matmul and GPU matmul disagree for {m}x{k}@{k}x{n}")
        log("  CPU matmul vs NumPy vs GPU checked")
        log(f"  {'n':>5} {'cpu matmul':>11} {'gpu total':>10} {'numpy':>8}   (ms)")
        for n in (64, 256, 512):
            A = rng.standard_normal((n, n)).astype(np.float32)
            B = rng.standard_normal((n, n)).astype(np.float32)
            ma, mb = mat(n, n, A), mat(n, n, B)
            try:
                tc = median_time(lambda: ma.matmul(mb), 1, 3)
            except Exception:
                break
            tg = median_time(lambda: gmm(ma, mb, False).to_list(), 1, 3)
            tn = median_time(lambda: A @ B, 1, 3)
            log(f"  {n:>5} {tc * 1e3:>11.2f} {tg * 1e3:>10.2f} {tn * 1e3:>8.2f}")
    else:
        res.skip("no PyMatrix.matmul (CPU path)")
    if hasattr(M, "add"):
        for (r, c) in [(1, 1), (7, 9), (100, 257)]:
            A = rng.standard_normal((r, c)).astype(np.float32)
            B = rng.standard_normal((r, c)).astype(np.float32)
            if not np.allclose(to_np(mat(r, c, A).add(mat(r, c, B)), r, c), A + B, rtol=1e-5, atol=1e-6):
                res.fail(f"add wrong for {r}x{c}")
        probe(res, "add shape mismatch", lambda: mat(2, 3, range(6)).add(mat(3, 2, range(6))).to_list(), "fail")
        log("  add checked")


def t_dispatch_overhead(res, a):
    P = E().PyMatrix
    rng = np.random.default_rng(SEED)
    n = 64
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    la = A.ravel().tolist()
    ma, mb, one = mat(n, n, A), mat(n, n, B), mat(1, 1, [1.0])
    mc = gmm(ma, mb, False)
    reps = 30 if a.quick else 100
    ops = [("gpu_matmul 1x1 (pure call overhead)", lambda: one.gpu_matmul(one, use_relu=False)),
           ("gpu_matmul 64x64 + relu", lambda: ma.gpu_matmul(mb, use_relu=True)),
           ("gpu_matmul 64x64 plain", lambda: ma.gpu_matmul(mb, use_relu=False)),
           ("multiply 64x64", lambda: ma.multiply(mb)),
           ("add 64x64", lambda: ma.add(mb)),
           ("sub 64x64", lambda: ma.sub(mb)),
           ("scale 64x64", lambda: ma.scale(0.5)),
           ("transpose 64x64", lambda: ma.transpose()),
           ("from_list 64x64", lambda: P.from_list(n, n, la)),
           ("to_list 64x64", lambda: mc.to_list())]
    log("  median latency per call (ms); GPU ops are synchronous if these are all ~equal and non-trivial")
    times = {}
    for name, fn in ops:
        try:
            times[name] = median_time(fn, 5, reps) * 1e3
            log(f"  {name:<40} {times[name]:>8.3f}")
        except Exception as e:
            log(f"  {name:<40} skipped ({type(e).__name__})")
    base = times.get("gpu_matmul 1x1 (pure call overhead)")
    elem = [times[k] for k in ("multiply 64x64", "add 64x64", "sub 64x64", "scale 64x64") if k in times]
    if base is not None and base > 0.4:
        res.warn(f"fixed cost per gpu_matmul call is {base:.2f} ms even for a 1x1 matrix (no work): "
                 f"buffer/bind-group creation, submit and a blocking poll/readback per call")
    if base is not None and elem and max(elem) < 0.05 and base > 0.4:
        res.warn("multiply/add/sub/scale take ~1 microsecond: they are almost certainly HOST-side (CPU) ops, "
                 "so PyMatrix data lives on the host and every gpu_matmul does upload + dispatch + readback; "
                 "'GPU-resident' chaining is not actually device-resident")


def _build_mlp(sizes):
    from network import SimpleMLP
    return SimpleMLP(sizes)


def t_mlp_learning(res, a):
    rng = np.random.default_rng(SEED)
    N, D = 8, 4
    X = rng.standard_normal((N, D)).astype(np.float32)
    y = (0.5 + 0.4 * np.tanh(X[:, 0] - 0.5 * X[:, 1])).reshape(N, 1).astype(np.float32)
    dead_loss = float(np.mean(y ** 2))              # loss when the output ReLU outputs exactly 0
    Xm = mat(N, D, X)
    epochs = 250 if a.quick else 500
    counts = {"learning": 0, "collapsed to zero output": 0, "stuck": 0, "not learning": 0, "diverged": 0}
    runs = 0
    # network.py: ReLU on EVERY layer incl. the output, grad = 2*(p-t) (sum, not mean) -> small lr, many seeds.
    for seed in range(1, 7):
        row = []
        for lr in (0.003, 0.001, 0.0003):
            random.seed(seed)
            mlp = _build_mlp([D, 16, 16, 1])
            losses = []
            for _ in range(epochs):
                p = to_np(mlp.forward(Xm), N, 1)
                loss = float(np.mean((p - y) ** 2))
                losses.append(loss)
                if not math.isfinite(loss) or loss > 1e6:
                    break
                dv = mat(N, 1, 2.0 * (p - y))
                for layer in reversed(mlp.layers):
                    dv = layer.backward(dv, learning_rate=lr)
            first, last = losses[0], losses[-1]
            if not all(math.isfinite(x) for x in losses) or last > 1e6:
                st = "diverged"
            elif max(losses) - min(losses) < 1e-9:
                st = "stuck"
            elif abs(last - dead_loss) < 1e-4 * max(dead_loss, 1.0):
                st = "collapsed to zero output"
            elif last < 0.6 * min(first, dead_loss):
                st = "learning"
            else:
                st = "not learning"
            counts[st] += 1
            runs += 1
            row.append(f"lr={lr}: {first:.3f}->{last:.4f} [{st}]")
        log(f"  seed={seed}  " + " | ".join(row))
    log("  summary over %d runs: " % runs + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    if counts["learning"] == 0:
        res.fail("no run learned: suspect a forward/backward/update bug")
    elif counts["learning"] < runs:
        res.warn(f"only {counts['learning']}/{runs} runs learned; "
                 f"{counts['collapsed to zero output'] + counts['stuck']} died at a zero output "
                 f"(ReLU on the output layer = dying-ReLU risk; use a linear output layer for regression)")


def t_stress_endurance(res, a):
    rng = np.random.default_rng(SEED)
    iters = 400 if a.quick else 2000
    n = 128
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    ref = ref_matmul(A, B, True)
    ma, mb = mat(n, n, A), mat(n, n, B)
    for _ in range(30):                                   # warm-up: device init, pipelines, driver pools
        gmm(ma, mb, True).to_list()
    batch = max(iters // 10, 1)
    times, mems = [], []
    for i in range(iters):
        if i % batch == 0:
            t0 = time.perf_counter()
        out = gmm(ma, mb, True)
        if i % 100 == 0 and not check_close(to_np(out, n, n), ref, n)[0]:
            res.fail(f"silent corruption at iteration {i}")
            return
        del out
        if (i + 1) % batch == 0:
            times.append((time.perf_counter() - t0) / batch)
            mems.append(rss_mb())
    log(f"  {iters} matmuls ({n}x{n}): per-iter ms first batch {times[0] * 1e3:.3f}, last {times[-1] * 1e3:.3f}")
    if times[-1] > 1.5 * times[0] + 1e-4:
        res.warn("per-iteration time drifts upward over the run")
    if mems[0] is None:
        res.skip("host memory not measured (pip install psutil)")
    else:
        log("  host RSS after each batch (MB): " + " ".join(f"{m:.0f}" for m in mems))
        half = len(mems) // 2
        slope = (mems[-1] - mems[half]) / ((len(mems) - 1 - half) * batch) * 1000
        log(f"  slope over the second half: {slope:+.1f} MB per 1000 matmuls")
        if slope > 200:
            res.fail(f"host memory keeps growing ({slope:.0f} MB/1000 matmuls): leak")
        elif slope > 10:
            res.warn(f"host memory still growing at {slope:.0f} MB/1000 matmuls after warm-up")
        lst = A.ravel().tolist()
        churn = 150 if a.quick else 500
        marks = [rss_mb()]
        for _ in range(2):
            for _ in range(churn):
                E().PyMatrix.from_list(n, n, lst)
            marks.append(rss_mb())
        g1, g2 = marks[1] - marks[0], marks[2] - marks[1]
        log(f"  create/destroy churn: round 1 {g1:+.0f} MB, round 2 {g2:+.0f} MB ({churn} matrices each)")
        if g2 > 150:
            res.fail(f"matrix create/destroy leaks ~{g2:.0f} MB per {churn} matrices")
        elif g2 > 30 and g2 > 0.5 * g1:
            res.warn("matrix create/destroy keeps growing in round 2 as well: probable leak (pool would plateau)")
    log("  note: only host RAM is observed here. Watch VRAM in Task Manager / nvidia-smi while it runs.")


def t_benchmark(res, a):
    rng = np.random.default_rng(SEED)
    sizes = [64, 128, 256, 512] if a.quick else [64, 128, 256, 512, 1024]
    if a.big:
        sizes.append(2048)
    log("  all times in ms, medians; total = matmul + readback (forces GPU sync)")
    log(f"  {'n':>5} {'np->list':>9} {'upload':>8} {'matmul':>8} {'readback':>9} {'total':>9} "
        f"{'GFLOPS':>8} {'numpy':>8} {'np GF':>7} {'relu x':>7}")
    for n in sizes:
        reps = 3 if n >= 1024 else 7
        A = rng.standard_normal((n, n), dtype=np.float32)
        B = rng.standard_normal((n, n), dtype=np.float32)
        la, lb = A.ravel().tolist(), B.ravel().tolist()
        P = E().PyMatrix
        t_conv = median_time(lambda: A.ravel().tolist(), 1, reps)
        t_up = median_time(lambda: P.from_list(n, n, la), 1, reps)
        a_, b_ = P.from_list(n, n, la), P.from_list(n, n, lb)
        t_mm = median_time(lambda: a_.gpu_matmul(b_, use_relu=False), 2, reps)
        c_ = a_.gpu_matmul(b_, use_relu=False)
        t_dl = median_time(lambda: c_.to_list(), 1, reps)
        t_tot = median_time(lambda: a_.gpu_matmul(b_, use_relu=False).to_list(), 2, reps)
        t_tot_r = median_time(lambda: a_.gpu_matmul(b_, use_relu=True).to_list(), 2, reps)
        t_np = median_time(lambda: A @ B, 1, reps)
        flops = 2.0 * n ** 3
        ratio = t_tot_r / t_tot
        log(f"  {n:>5} {t_conv * 1e3:>9.2f} {t_up * 1e3:>8.2f} {t_mm * 1e3:>8.2f} {t_dl * 1e3:>9.2f} "
            f"{t_tot * 1e3:>9.2f} {flops / t_tot / 1e9:>8.1f} {t_np * 1e3:>8.2f} {flops / t_np / 1e9:>7.1f} "
            f"{ratio:>7.2f}")
        if n >= 256 and ratio > 1.3:
            res.warn(f"n={n}: fused ReLU costs {100 * (ratio - 1):.0f}% extra vs plain matmul "
                     f"(a fused epilogue should be near 0%)")
        if n >= 1024 and t_tot > t_np:
            res.warn(f"n={n}: GPU path (incl. transfers) is slower than NumPy on CPU")
    # chaining: does data stay on the GPU between ops?
    n, steps = 256, 8
    X0 = rng.standard_normal((n, n), dtype=np.float32)
    W = (rng.standard_normal((n, n)) / math.sqrt(n)).astype(np.float32)
    P = E().PyMatrix
    x0, w = mat(n, n, X0), mat(n, n, W)

    def chain_gpu():
        x = x0
        for _ in range(steps):
            x = x.gpu_matmul(w, use_relu=True)
        x.to_list()

    def chain_roundtrip():
        x = x0
        for _ in range(steps):
            x = x.gpu_matmul(w, use_relu=True)
            x = P.from_list(n, n, x.to_list())

    t_a, t_b = median_time(chain_gpu, 1, 5), median_time(chain_roundtrip, 1, 5)
    log(f"  chain of {steps} matmuls @ {n}: no to_list/from_list between steps {t_a * 1e3:.1f} ms vs with them each step "
        f"{t_b * 1e3:.1f} ms (x{t_b / t_a:.2f})")
    if t_b / t_a < 1.15:
        res.warn("skipping to_list/from_list between ops gives ~no speed-up: results are probably copied back "
                 "to the host after every op anyway, or transfers are not the bottleneck")


def t_mlp_profile(res, a):
    rng = np.random.default_rng(SEED)
    B = 128 if a.quick else 256
    sizes = [64, 128, 128, 1]
    X = rng.standard_normal((B, sizes[0])).astype(np.float32)
    y = np.abs(X[:, :1]).astype(np.float32)
    Xm = mat(B, sizes[0], X)
    mlp = _build_mlp(sizes)
    warm, reps = 3, 15 if a.quick else 30
    tf, tl, tb, tm = [], [], [], []
    for ep in range(warm + reps):
        t0 = time.perf_counter()
        pred = mlp.forward(Xm)
        t1 = time.perf_counter()
        p = to_np(pred, B, 1)                       # readback + loss, as any real loop needs
        dv = mat(B, 1, 2.0 * (p - y))
        t2 = time.perf_counter()
        for layer in reversed(mlp.layers):
            dv = layer.backward(dv, learning_rate=0.001)
        t3 = time.perf_counter()
        if ep >= warm:
            if all(hasattr(l, "output") for l in mlp.layers):
                t4 = time.perf_counter()
                for l in mlp.layers:                     # same host round trip DenseLayer.backward does
                    ol = l.output.to_list()
                    g = [1.0 if v > 0.0 else 0.0 for v in ol]
                    E().PyMatrix.from_list(l.output.rows, l.output.cols, g)
                tm.append(time.perf_counter() - t4)
            tf.append(t1 - t0)
            tl.append(t2 - t1)
            tb.append(t3 - t2)
    f, l, b = (statistics.median(x) * 1e3 for x in (tf, tl, tb))
    tot = f + l + b
    log(f"  MLP {sizes}, batch {B}: forward {f:.2f} ms ({100 * f / tot:.0f}%), loss+readback {l:.2f} ms "
        f"({100 * l / tot:.0f}%), backward+update {b:.2f} ms ({100 * b / tot:.0f}%), epoch {tot:.2f} ms")
    if not np.all(np.isfinite(p)):
        res.fail("non-finite predictions during profiling")
    if tm:
        m = statistics.median(tm) * 1e3
        log(f"  ReLU-derivative mask built on CPU (to_list + Python loop + from_list) per epoch: "
            f"{m:.2f} ms = {100 * m / b:.0f}% of backward, {100 * m / tot:.0f}% of the epoch")
        if m / b > 0.3:
            res.warn("host round trip for the ReLU mask dominates backward: add a GPU kernel that fuses "
                     "dz = dvalues * (out > 0) (a relu_backward) and keep the mask on the device")
    if b > 3 * f:
        res.warn("backward+update is >3x forward: look for per-layer transposes/readbacks inside backward")


TESTS = {
    "api_surface": (t_api_surface, "interface discovery"),
    "roundtrip": (t_roundtrip, "from_list/to_list + row-major layout"),
    "matmul_correctness": (t_matmul_correctness, "gpu_matmul vs NumPy, odd shapes, relu on/off"),
    "relu_extremes": (t_relu_extremes, "ReLU semantics on extreme values"),
    "transpose": (t_transpose, "transpose correctness + involution"),
    "special_values": (t_special_values, "NaN / Inf / overflow behaviour"),
    "invalid_inputs": (t_invalid_inputs, "bad shapes/types: clean error vs panic vs silent garbage"),
    "elementwise_ops": (t_elementwise_ops, "constructor, multiply, sub, scale, default matmul semantics"),
    "cpu_gpu_parity": (t_cpu_gpu_parity, "CPU matmul vs GPU vs NumPy, add"),
    "state_isolation": (t_state_isolation, "buffer aliasing, input mutation, chaining, determinism"),
    "fuzz_shapes": (t_fuzz_shapes, "random shapes and magnitudes"),
    "layer_backward": (t_layer_backward, "gradient check, stale-weights check, bias usage"),
    "mlp_learning": (t_mlp_learning, "end-to-end: does backward actually learn?"),
    "stress_endurance": (t_stress_endurance, "long loop: corruption, drift, host memory growth"),
    "dispatch_overhead": (t_dispatch_overhead, "per-call latency of every op, adapter info"),
    "benchmark": (t_benchmark, "throughput, transfer split, ReLU fusion cost, chaining"),
    "mlp_profile": (t_mlp_profile, "forward / readback / backward breakdown"),
}


# ----------------------------------------------------------------------------- runner
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
    for n in names:
        if n not in TESTS:
            sys.exit(f"unknown test '{n}'. use --list")
    log("=" * 74)
    log(" ml_core_engine independent black-box audit")
    log("=" * 74)
    summary = []
    for name in names:
        log(f"\n[{name}] {TESTS[name][1]}")
        cmd = [sys.executable, os.path.abspath(__file__), "--child", name]
        if args.quick:
            cmd.append("--quick")
        if args.big:
            cmd.append("--big")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
        timed_out = []
        timer = threading.Timer(args.timeout, lambda: (timed_out.append(1), proc.kill()))
        timer.start()
        result = None
        try:
            for line in proc.stdout:
                if line.startswith(RESULT_TAG):
                    result = json.loads(line[len(RESULT_TAG):])
                else:
                    sys.stdout.write(line)
                    sys.stdout.flush()
        finally:
            timer.cancel()
            proc.wait()
        if result is None:
            code = proc.returncode & 0xFFFFFFFF
            why = "timed out" if timed_out else f"process died hard (exit 0x{code:08X}): Rust abort / wgpu error / driver crash"
            result = {"status": FAIL, "notes": ["FAIL: " + why]}
            log("  [FAIL] " + why)
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated test names")
    ap.add_argument("--quick", action="store_true", help="smaller sizes / fewer iterations")
    ap.add_argument("--big", action="store_true", help="include 2048 benchmark (RAM hungry)")
    ap.add_argument("--timeout", type=int, default=900, help="per-test timeout in seconds")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.list:
        for n, (_, d) in TESTS.items():
            print(f"{n:<20} {d}")
        return
    if args.child:
        run_child(args.child, args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()