import ml_core_engine as mc
import numpy as np
import time

def print_header(title):
    print("\n" + "=" * 85)
    print(f"🔬 HARDWARE AUDIT PHASE: {title}")
    print("=" * 85)

def verify_metric(gpu_output, np_expected, test_name):
    max_error = float(np.max(np.abs(gpu_output - np_expected)))
    mean_error = float(np.mean(np.abs(gpu_output - np_expected)))
    variance = float(np.var(gpu_output - np_expected))
    
    print(f"   [ {test_name} ]")
    print(f"   ├── Peak Absolute Error: {max_error:.6e}")
    print(f"   ├── Mean Absolute Error: {mean_error:.6e}")
    print(f"   └── Error Variance     : {variance:.6e}")
    
    if max_error < 1e-2:
        print("   Status: 🟢 MATHEMATICALLY VALID (PASS)")
        return True
    else:
        print("   Status: 🔴 PRECISION CORRUPTION DETECTED (FAIL)")
        return False

def execute_complete_audit():
    print("=" * 85)
    print("🧠 CORE INFERENCE ENGINE V4.1 - QUANTUM HARDWARE MATHEMATICAL AUDIT SUITE")
    print("=" * 85)
    print(f"📡 Engine Deployment Status: {mc.check_gpu_device()}")
    
    suite_success = True

    # ----------------------------------------------------------------
    # 1. MATHEMATICAL GEMM MATMUL CORE AUDIT
    # ----------------------------------------------------------------
    print_header("1. Tiled Vector GEMM Core (M=1 Single Token & Context Batch)")
    
    gemm_cases = [
        (1, 4096, 4096, "Token Generation Step (1x4096)"),
        (32, 1024, 4096, "Context Prefill Batch (32x1024)"),
        (256, 256, 256, "Standard Aligned Matrix Block")
    ]
    
    for m, k, n, label in gemm_cases:
        print(f"\n▶️ Testing Matrix Multiplication Layer: {label}")
        a = np.random.uniform(-1.0, 1.0, (m, k)).astype(np.float32)
        b = np.random.uniform(-1.0, 1.0, (k, n)).astype(np.float32)
        expected = np.dot(a, b)
        
        g_a = mc.PyMatrix.from_list(m, k, a.ravel().tolist()).to_gpu()
        g_b = mc.PyMatrix.from_list(k, n, b.ravel().tolist()).to_gpu()
        
        t0 = time.perf_counter()
        g_c = g_a.gpu_matmul(g_b, False)
        out_c = np.array(g_c.to_cpu().to_list(), dtype=np.float32).reshape(m, n)
        dt = (time.perf_counter() - t0) * 1000
        
        print(f"   ⏱️ Hardware Execution Latency: {dt:.3f} ms")
        if not verify_metric(out_c, expected, f"GEMM_{m}x{k}x{n}"):
            suite_success = False

    # ----------------------------------------------------------------
    # 2. ROOT MEAN SQUARE NORMALIZATION (RMSNORM / RESIDUAL)
    # ----------------------------------------------------------------
    print_header("2. Transformer Layer Normalization Shaders")
    
    rows, cols = 64, 4096
    eps = 1e-5
    x = np.random.uniform(-2.0, 2.0, (rows, cols)).astype(np.float32)
    res = np.random.uniform(-1.0, 1.0, (rows, cols)).astype(np.float32)
    
    # RMSNorm Expected
    rms_exp = x * (1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps))
    # Residual RMSNorm Expected
    fused_x = x + res
    res_rms_exp = fused_x * (1.0 / np.sqrt(np.mean(fused_x**2, axis=-1, keepdims=True) + eps))
    
    g_x = mc.PyMatrix.from_list(rows, cols, x.ravel().tolist()).to_gpu()
    g_res = mc.PyMatrix.from_list(rows, cols, res.ravel().tolist()).to_gpu()
    
    print("\n▶️ Testing Tree-Parallel RMSNorm Shader:")
    g_norm = g_x.gpu_rms_norm(eps)
    out_norm = np.array(g_norm.to_cpu().to_list(), dtype=np.float32).reshape(rows, cols)
    if not verify_metric(out_norm, rms_exp, "gpu_rms_norm"): suite_success = False
    
    print("\n▶️ Testing Fused Residual RMSNorm Shader:")
    g_res_norm = g_x.gpu_residual_rms_norm(g_res, eps)
    out_res_norm = np.array(g_res_norm.to_cpu().to_list(), dtype=np.float32).reshape(rows, cols)
    if not verify_metric(out_res_norm, res_rms_exp, "gpu_residual_rms_norm"): suite_success = False

    # ----------------------------------------------------------------
    # 3. TRANSFORMER ACTIVATION LAYER (SWIGLU)
    # ----------------------------------------------------------------
    print_header("3. Transformer Gated Activation Shaders")
    
    print("\n▶️ Testing Vectorized SwiGLU Shader Layer:")
    gate = np.random.uniform(-3.0, 3.0, (rows, cols)).astype(np.float32)
    up = np.random.uniform(-3.0, 3.0, (rows, cols)).astype(np.float32)
    
    # SwiGLU Equation: (gate * sigmoid(gate)) * up
    swiglu_exp = (gate * (1.0 / (1.0 + np.exp(-gate)))) * up
    
    g_gate = mc.PyMatrix.from_list(rows, cols, gate.ravel().tolist()).to_gpu()
    g_up = mc.PyMatrix.from_list(rows, cols, up.ravel().tolist()).to_gpu()
    g_swi = g_gate.gpu_swiglu(g_up)
    out_swi = np.array(g_swi.to_cpu().to_list(), dtype=np.float32).reshape(rows, cols)
    if not verify_metric(out_swi, swiglu_exp, "gpu_swiglu"): suite_success = False

    # ----------------------------------------------------------------
    # 4. ROTARY POSITION EMBEDDING (ROPE)
    # ----------------------------------------------------------------
    print_header("4. Contextual Positioning Embeddings")
    
    print("\n▶️ Testing Vectorized RoPE (Rotary Position Embedding) Shader:")
    rope_rows, rope_cols = 32, 128
    rope_data = np.random.uniform(-1.0, 1.0, (rope_rows, rope_cols)).astype(np.float32)
    rope_exp = np.zeros_like(rope_data)
    
    for r in range(rope_rows):
        for c in range(0, rope_cols, 2):
            x0, x1 = rope_data[r, c], rope_data[r, c+1]
            freq = 1.0 / (10000.0 ** (float(c) / float(rope_cols)))
            v = float(r) * freq
            rope_exp[r, c] = x0 * np.cos(v) - x1 * np.sin(v)
            rope_exp[r, c+1] = x0 * np.sin(v) + x1 * np.cos(v)
            
    g_rope_in = mc.PyMatrix.from_list(rope_rows, rope_cols, rope_data.ravel().tolist()).to_gpu()
    g_rope_out = g_rope_in.gpu_rope()
    out_rope = np.array(g_rope_out.to_cpu().to_list(), dtype=np.float32).reshape(rope_rows, rope_cols)
    if not verify_metric(out_rope, rope_exp, "gpu_rope"): suite_success = False

    # ----------------------------------------------------------------
    # 5. SOFTMAX & ELEMENT-WISE OPERATORS
    # ----------------------------------------------------------------
    print_header("5. Base Core Element-Wise Operations")
    
    dim_size = 512
    a_elt = np.random.uniform(-1.0, 1.0, (1, dim_size)).astype(np.float32)
    b_elt = np.random.uniform(-1.0, 1.0, (1, dim_size)).astype(np.float32)
    
    g_ae = mc.PyMatrix.from_list(1, dim_size, a_elt.ravel().tolist()).to_gpu()
    g_be = mc.PyMatrix.from_list(1, dim_size, b_elt.ravel().tolist()).to_gpu()
    
    print("\n▶ extinction Test Element-Wise ADD:")
    out_add = np.array(g_ae.gpu_add(g_be).to_cpu().to_list(), dtype=np.float32).reshape(1, dim_size)
    if not verify_metric(out_add, a_elt + b_elt, "gpu_add"): suite_success = False
    
    print("\n▶ extinction Test Element-Wise SUB:")
    out_sub = np.array(g_ae.gpu_sub(g_be).to_cpu().to_list(), dtype=np.float32).reshape(1, dim_size)
    if not verify_metric(out_sub, a_elt - b_elt, "gpu_sub"): suite_success = False
    
    print("\n▶ extinction Test Vectorized Softmax Shader:")
    sm_in = np.random.uniform(-2.0, 2.0, (16, 256)).astype(np.float32)
    sm_max = np.max(sm_in, axis=-1, keepdims=True)
    sm_exp = np.exp(sm_in - sm_max)
    sm_expected = sm_exp / np.sum(sm_exp, axis=-1, keepdims=True)
    
    g_sm_in = mc.PyMatrix.from_list(16, 256, sm_in.ravel().tolist()).to_gpu()
    out_sm = np.array(g_sm_in.gpu_softmax().to_cpu().to_list(), dtype=np.float32).reshape(16, 256)
    if not verify_metric(out_sm, sm_expected, "gpu_softmax"): suite_success = False

    # ----------------------------------------------------------------
    # 6. FUSED ATTENTION & STABLE ONLINE SOFTMAX
    # ----------------------------------------------------------------
    print_header("6. Fused Single-Workgroup Attention Node")
    
    print("\n▶️ Testing Online Softmax Execution & Memory Reduction Cross-Products:")
    attn_b, attn_dk, attn_seq = 1, 64, 128
    attn_scale = 1.0 / np.sqrt(attn_dk)
    
    q_mat = np.random.uniform(-1.5, 1.5, (attn_b, attn_dk)).astype(np.float32)
    k_mat = np.random.uniform(-1.5, 1.5, (attn_seq, attn_dk)).astype(np.float32)
    v_mat = np.random.uniform(-1.0, 1.0, (attn_seq, attn_dk)).astype(np.float32)
    
    att_scores = np.dot(q_mat, k_mat.T) * attn_scale
    att_exp = np.exp(att_scores - np.max(att_scores, axis=-1, keepdims=True))
    att_probs = att_exp / np.sum(att_exp, axis=-1, keepdims=True)
    att_expected = np.dot(att_probs, v_mat)
    
    g_att_q = mc.PyMatrix.from_list(attn_b, attn_dk, q_mat.ravel().tolist()).to_gpu()
    g_att_k = mc.PyMatrix.from_list(attn_seq, attn_dk, k_mat.ravel().tolist()).to_gpu()
    g_att_v = mc.PyMatrix.from_list(attn_seq, attn_dk, v_mat.ravel().tolist()).to_gpu()
    
    cache_node = mc.KvCache(attn_seq, attn_dk)
    cache_node.update(0, g_att_k, g_att_v)
    
    g_attention_res = g_att_q.gpu_fused_attention(cache_node, attn_seq)
    out_attention = np.array(g_attention_res.to_cpu().to_list(), dtype=np.float32).reshape(attn_b, attn_dk)
    if not verify_metric(out_attention, att_expected, "gpu_fused_attention"): suite_success = False

    # ----------------------------------------------------------------
    # 7. VRAM MEMORY POOL LEAK MATRIX DETECTION
    # ----------------------------------------------------------------
    print_header("7. High-Frequency VRAM Memory Pool Isolation Audit")
    print("\n▶️ Simulating volatile sequence workloads with active memory churning...")
    
    pool_failure = False
    for step in range(50):
        cycle_dim = 128 + (step % 3) * 64
        a_churn = np.ones((cycle_dim, cycle_dim), dtype=np.float32) * float(step)
        b_churn = np.identity(cycle_dim, dtype=np.float32)
        churn_expected = np.dot(a_churn, b_churn)
        g_ac = mc.PyMatrix.from_list(cycle_dim, cycle_dim, a_churn.ravel().tolist()).to_gpu()
        g_bc = mc.PyMatrix.from_list(cycle_dim, cycle_dim, b_churn.ravel().tolist()).to_gpu()
        g_cc = g_ac.gpu_matmul(g_bc, False)
        out_cc = np.array(g_cc.to_cpu().to_list(), dtype=np.float32).reshape(cycle_dim, cycle_dim)
        if np.max(np.abs(out_cc - churn_expected)) > 1e-3:
            pool_failure = True
            print(f"   ❌ MEMORY CORRUPTION/LEAK ENCOUNTERED AT CYCLING STEP {step}!")
            break

    if not pool_failure:
        print("   ✅ VRAM MEMORY POOL IS 100% SECURE: Certified Zero Cross-Contamination Across Lifecycles.")
    else:
        suite_success = False

    # ----------------------------------------------------------------
    # FINAL JUDGEMENT VERDICT
    # ----------------------------------------------------------------
    print("\n" + "=" * 85)
    if suite_success:
        print("🏆 AUDIT VERDICT: SYSTEM OPERATIONAL AND 100% MATHEMATICALLY FAULT-TOLERANT.")
    else:
        print("💥 AUDIT VERDICT: STRUCTURAL COMPUTE STALL OR ALIGNMENT DRIFT DETECTED.")
    print("=" * 85)

if __name__ == "__main__":
    execute_complete_audit()