import ml_core_engine

def run_comprehensive_test():
    print("=================================================================")
    print("🚀 Starting End-to-End Comprehensive Test for ML Core GPU Engine")
    print("=================================================================")

    print("\n[1/7] Checking GPU initialization via Persistent Context...")
    print(ml_core_engine.check_gpu_device())

    print("\n[2/7] Testing Advanced Matrix Multiplication (gpu_matmul + ReLU)...")
    a = ml_core_engine.PyMatrix.from_list(2, 2, [1.0, -2.0, 3.0, 4.0]).to_gpu()
    b = ml_core_engine.PyMatrix.from_list(2, 2, [0.5, 1.0, -1.0, 2.0]).to_gpu()
    res_matmul_gpu = a.gpu_matmul(b, use_relu=True)
    res_matmul = res_matmul_gpu.to_cpu()
    print("✨ MatMul + ReLU Result:", res_matmul.to_list())
    assert len(res_matmul.to_list()) == 4, "Error in MatMul output dimensions"

    print("\n[3/7] Testing Root Mean Square Normalization (gpu_rms_norm)...")
    mat_rms = ml_core_engine.PyMatrix.from_list(2, 4, [1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0]).to_gpu()
    res_rms_gpu = mat_rms.gpu_rms_norm(1e-5)
    res_rms = res_rms_gpu.to_cpu()
    print("✨ RMSNorm Result:", res_rms.to_list())
    assert res_rms.rows == 2 and res_rms.cols == 4, "Error in RMSNorm dimensions"

    print("\n[4/7] Testing SwiGLU Hybrid Activation Function (gpu_swiglu)...")
    gate = ml_core_engine.PyMatrix.from_list(1, 4, [1.0, 0.0, -1.0, 2.0]).to_gpu()
    up = ml_core_engine.PyMatrix.from_list(1, 4, [2.0, 3.0, 4.0, 5.0]).to_gpu()
    res_swiglu_gpu = gate.gpu_swiglu(up)
    res_swiglu = res_swiglu_gpu.to_cpu()
    print("✨ SwiGLU Result:", res_swiglu.to_list())
    assert len(res_swiglu.to_list()) == 4, "Error in SwiGLU output"

    print("\n[5/7] Testing Softmax Probability Distribution (gpu_softmax)...")
    mat_soft = ml_core_engine.PyMatrix.from_list(2, 3, [1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).to_gpu()
    res_soft_gpu = mat_soft.gpu_softmax()
    res_soft = res_soft_gpu.to_cpu()
    print("✨ Softmax Result:", res_soft.to_list())
    
    list_soft = res_soft.to_list()
    row1_sum = sum(list_soft[0:3])
    row2_sum = sum(list_soft[3:6])
    print(f"   -> Row 1 Sum: {row1_sum:.4f} | Row 2 Sum: {row2_sum:.4f}")
    assert abs(row1_sum - 1.0) < 1e-5 and abs(row2_sum - 1.0) < 1e-5, "Error in Softmax probabilities sum"

    print("\n[6/7] Testing Rotary Position Embedding (gpu_rope)...")
    mat_rope = ml_core_engine.PyMatrix.from_list(2, 4, [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]).to_gpu()
    res_rope_gpu = mat_rope.gpu_rope()
    res_rope = res_rope_gpu.to_cpu()
    print("✨ RoPE Result:", res_rope.to_list())
    assert len(res_rope.to_list()) == 8, "Error in RoPE output"

    print("\n[7/7] Testing Scaled Dot-Product Attention (gpu_attention)...")
    query = ml_core_engine.PyMatrix.from_list(2, 2, [1.0, 0.0, 0.0, 1.0]).to_gpu()
    key = ml_core_engine.PyMatrix.from_list(2, 2, [1.0, 0.0, 0.0, 1.0]).to_gpu()
    value = ml_core_engine.PyMatrix.from_list(2, 2, [10.0, 20.0, 30.0, 40.0]).to_gpu()
    res_attn_gpu = query.gpu_attention(key, value)
    res_attn = res_attn_gpu.to_cpu()
    print("✨ Attention Result:", res_attn.to_list())
    assert res_attn.rows == 2 and res_attn.cols == 2, "Error in Attention output dimensions"

    print("\n=================================================================")
    print("🎉 ML Core Engine is now fully complete with Attention mechanism! 🦾🔥")
    print("=================================================================")

if __name__ == "__main__":
    run_comprehensive_test()
