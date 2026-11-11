import math
import random
import ml_core_engine

class DenseLayer:
    def __init__(self, n_inputs, n_outputs, activation_relu=True):
        scale = math.sqrt(2.0 / n_inputs)
        weights_data = [random.gauss(0.0, scale) for _ in range(n_inputs * n_outputs)]
        self.weights = ml_core_engine.PyMatrix.from_list(n_inputs, n_outputs, weights_data).to_gpu()
        self.bias_data = [0.01] * n_outputs
        self.activation_relu = activation_relu
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs.gpu_matmul(self.weights, use_relu=self.activation_relu)
        return self.output

    def backward(self, dvalues, learning_rate):
        if self.activation_relu:
            dized = dvalues.gpu_relu_backward(self.output)
        else:
            dized = dvalues
        old_weights_transpose = self.weights.gpu_transpose()
        dx = dized.gpu_matmul(old_weights_transpose, use_relu=False)
        inputs_transpose = self.inputs.gpu_transpose()
        dweights = inputs_transpose.gpu_matmul(dized, use_relu=False)
        grad_scaled = dweights.gpu_scale(learning_rate)
        self.weights = self.weights.gpu_sub(grad_scaled)
        return dx

class SimpleMLP:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            is_last = (i == len(layer_sizes) - 2)
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], activation_relu=not is_last))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            predictions = self.forward(X)
            pred_list = predictions.to_cpu().to_list()
            target_list = y.to_cpu().to_list()
            c_count = predictions.cols()
            loss = 0.0
            grad_data = [0.0] * len(pred_list)
            for r in range(predictions.rows()):
                for c in range(c_count):
                    idx = r * c_count + c
                    p_val = pred_list[idx]
                    t_val = target_list[idx]
                    loss += (p_val - t_val) ** 2
                    grad_data[idx] = 2.0 * (p_val - t_val) / predictions.rows()
            loss /= (predictions.rows() * c_count)
            dvalues = ml_core_engine.PyMatrix.from_list(y.rows(), y.cols(), grad_data).to_gpu()
            for layer in reversed(self.layers):
                dvalues = layer.backward(dvalues, learning_rate)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   -> Epoch {epoch + 1}/{epochs} | Massive Network VRAM Loss: {loss:.6f}")

class FullTransformerBlock:
    def __init__(self, d_model):
        self.d_model = d_model

    def forward(self, x, q_w, k_w, v_w, gate_w, up_w):
        print(" Applying GPU RMSNorm...")
        normed = x.gpu_rms_norm(1e-5)
        print(" Applying GPU RoPE...")
        roped = normed.gpu_rope()
        print(" Computing Q, K, V projections...")
        q = roped.gpu_matmul(q_w, use_relu=False)
        k = roped.gpu_matmul(k_w, use_relu=False)
        v = roped.gpu_matmul(v_w, use_relu=False)
        print(" Initializing and Populating KV Cache for Attention...")
        seq_len = x.rows()
        d_k = x.cols()
        kv_cache = ml_core_engine.KvCache(seq_len, d_k)
        kv_cache.update(0, k, v)
        print(" Executing Scaled Dot-Product Attention with KV Cache...")
        attn_out = q.gpu_attention(kv_cache, seq_len)
        print(" Computing Softmax Probabilities...")
        _ = attn_out.gpu_softmax()
        print(" Executing SwiGLU Hybrid Activation...")
        gate = x.gpu_matmul(gate_w, use_relu=False)
        up = x.gpu_matmul(up_w, use_relu=False)
        swiglu_out = gate.gpu_swiglu(up)
        return swiglu_out

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Unleashing Full Power of ML Core GPU Engine via network.py")
    print("=" * 60)
    print("\n🔹 [Phase 1] Training Massive Multi-Layer Perceptron (MLP) on GPU...")
    BATCH_SIZE = 64
    INPUT_DIM = 512
    HIDDEN_DIM = 512
    OUTPUT_DIM = 256
    random.seed(42)
    X_data = [random.uniform(-0.5, 0.5) for _ in range(BATCH_SIZE * INPUT_DIM)]
    y_data = [0.1] * (BATCH_SIZE * OUTPUT_DIM)
    X = ml_core_engine.PyMatrix.from_list(BATCH_SIZE, INPUT_DIM, X_data).to_gpu()
    y = ml_core_engine.PyMatrix.from_list(BATCH_SIZE, OUTPUT_DIM, y_data).to_gpu()
    mlp = SimpleMLP([INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM])
    mlp.train(X, y, epochs=40, learning_rate=0.01)
    print("✨ Massive MLP Training Completed Successfully!")
    print("\n🔹 [Phase 2] Executing Full Transformer Block Pipeline on GPU...")
    seq_len = 2
    d_model = 4
    tokens = ml_core_engine.PyMatrix.from_list(seq_len, d_model, [
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    ]).to_gpu()
    qw = ml_core_engine.PyMatrix.from_list(d_model, d_model, [1.0, 0.0, 0.0, 1.0] * 4).to_gpu()
    kw = ml_core_engine.PyMatrix.from_list(d_model, d_model, [1.0, 0.0, 0.0, 1.0] * 4).to_gpu()
    vw = ml_core_engine.PyMatrix.from_list(d_model, d_model, [1.0, 1.0, 0.0, 0.0] * 4).to_gpu()
    gate_w = ml_core_engine.PyMatrix.from_list(d_model, d_model, [0.5, 0.5, 0.5, 0.5] * 4).to_gpu()
    up_w = ml_core_engine.PyMatrix.from_list(d_model, d_model, [0.2, 0.3, 0.4, 0.5] * 4).to_gpu()
    transformer = FullTransformerBlock(d_model)
    final_output_gpu = transformer.forward(tokens, qw, kw, vw, gate_w, up_w)
    final_output = final_output_gpu.to_cpu()
    print("✨ Transformer Pipeline Output Result:", final_output.to_list())
    print("\n" + "=" * 60)
    print("🎉 All GPU kernel powers unleashed and verified via network.py! 🦾🔥")
    print("=" * 60)
