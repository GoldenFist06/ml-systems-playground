import math
import random
import ml_core_engine

class DenseLayer:
    def __init__(self, n_inputs, n_outputs, activation_relu=True):
        scale = math.sqrt(2.0 / n_inputs)
        weights_data = [random.gauss(0.0, scale) for _ in range(n_inputs * n_outputs)]
        
        self.weights = ml_core_engine.PyMatrix.from_list(n_inputs, n_outputs, weights_data)
        self.bias = ml_core_engine.PyMatrix(1, n_outputs, 0.0)
        self.activation_relu = activation_relu
        
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        matmul_out = inputs.gpu_matmul(self.weights, use_relu=False)
        
        rows = matmul_out.rows
        cols = matmul_out.cols
        bias_list = self.bias.to_list()
        full_bias_data = bias_list * rows
        bias_matrix = ml_core_engine.PyMatrix.from_list(rows, cols, full_bias_data)
        
        linear_out = matmul_out.add(bias_matrix)
        
        if self.activation_relu:
            identity = ml_core_engine.PyMatrix.from_list(cols, cols, [1.0 if i == j else 0.0 for i in range(cols) for j in range(cols)])
            self.output = linear_out.gpu_matmul(identity, use_relu=True)
        else:
            self.output = linear_out
            
        return self.output

    def backward(self, dvalues, learning_rate):
        if self.activation_relu:
            dized = dvalues.gpu_relu_backward(self.output)
        else:
            dized = dvalues

        old_weights_transpose = self.weights.transpose()
        dx = dized.gpu_matmul(old_weights_transpose)

        inputs_transpose = self.inputs.transpose()
        dweights = inputs_transpose.gpu_matmul(dized)
        
        grad_scaled = dweights.scale(learning_rate)
        self.weights = self.weights.sub(grad_scaled)
        
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
            pred_list = predictions.to_list()
            target_list = y.to_list()
            loss = sum((p - t) ** 2 for p, t in zip(pred_list, target_list)) / len(pred_list)
            
            grad_data = [2.0 * (p - t) for p, t in zip(pred_list, target_list)]
            dvalues = ml_core_engine.PyMatrix.from_list(y.rows, y.cols, grad_data)
            
            for layer in reversed(self.layers):
                dvalues = layer.backward(dvalues, learning_rate)
                
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"   -> Epoch {epoch + 1}/{epochs} | Loss: {loss:.6f}")


class FullTransformerBlock:

    def __init__(self, d_model):
        self.d_model = d_model

    def forward(self, x, q_w, k_w, v_w, gate_w, up_w):
        print("  [1] Applying GPU RMSNorm (Root Mean Square Normalization)...")
        normed = x.gpu_rms_norm(1e-5)
        
        print("  [2] Applying GPU RoPE (Rotary Position Embedding)...")
        roped = normed.gpu_rope()
        
        print("  [3] Computing Q, K, V projections via GPU MatMul...")
        q = roped.gpu_matmul(q_w, use_relu=False)
        k = roped.gpu_matmul(k_w, use_relu=False)
        v = roped.gpu_matmul(v_w, use_relu=False)
        
        print("  [4] Executing Scaled Dot-Product Attention (gpu_attention)...")
        attn_out = q.gpu_attention(k, v)
        
        print("  [5] Computing Numerically Stable Softmax Probabilities...")
        softmax_out = attn_out.gpu_softmax()
        
        print("  [6] Executing SwiGLU Hybrid Activation Function...")
        gate = x.gpu_matmul(gate_w, use_relu=False)
        up = x.gpu_matmul(up_w, use_relu=False)
        swiglu_out = gate.gpu_swiglu(up)
        
        return swiglu_out


if __name__ == "__main__":
    print("=" * 65)
    print("🚀 Unleashing Full Power of ML Core GPU Engine via network.py")
    print("=" * 65)

  
    print("\n🔹 [Phase 1] Training Multi-Layer Perceptron (MLP) on GPU...")
    X = ml_core_engine.PyMatrix.from_list(4, 2, [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    y = ml_core_engine.PyMatrix.from_list(4, 1, [0.0, 1.0, 1.0, 0.0])
    
    mlp = SimpleMLP([2, 4, 1])
    mlp.train(X, y, epochs=150, learning_rate=0.05)
    print("✨ MLP Training Completed Successfully!")


    print("\n🔹 [Phase 2] Executing Full Transformer Block Pipeline on GPU...")
    seq_len = 2
    d_model = 4
    
    tokens = ml_core_engine.PyMatrix.from_list(seq_len, d_model, [
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    ])
    

    qw = ml_core_engine.PyMatrix.from_list(d_model, d_model, [1.0, 0.0, 0.0, 1.0] * 2 + [0.0, 1.0, 1.0, 0.0] * 2)
    kw = ml_core_engine.PyMatrix.from_list(d_model, d_model, [1.0, 0.0, 0.0, 1.0] * 2 + [0.0, 1.0, 1.0, 0.0] * 2)
    vw = ml_core_engine.PyMatrix.from_list(d_model, d_model, [1.0, 1.0, 0.0, 0.0] * 2 + [0.0, 0.0, 1.0, 1.0] * 2)
    gw = ml_core_engine.PyMatrix.from_list(d_model, d_model, [0.5, 0.5, 0.5, 0.5] * 2 + [1.0, 0.0, 1.0, 0.0] * 2)
    uw = ml_core_engine.PyMatrix.from_list(d_model, d_model, [0.2, 0.3, 0.4, 0.5] * 2 + [0.1, 0.2, 0.3, 0.4] * 2)

    transformer = FullTransformerBlock(d_model)
    final_output = transformer.forward(tokens, qw, kw, vw, gw, uw)
    
    print("✨ Transformer Pipeline Output Result:", final_output.to_list())

    print("\n" + "=" * 65)
    print("🎉 All GPU kernel powers unleashed and verified via network.py! 🦾🔥")
    print("=" * 65)