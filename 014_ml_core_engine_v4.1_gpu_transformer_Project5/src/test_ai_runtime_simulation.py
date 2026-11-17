import time
import sys
import random
import ml_core_engine as ml

class SovereignTransformerModel:
    def __init__(self, num_layers=4, d_model=256, n_heads=4, d_ff=256):
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.eps = 1e-5
        self.layers = []
        for _ in range(num_layers):
            layer = {
                "w_q": ml.PyMatrix(d_model, self.d_k, 0.02).to_gpu(),
                "w_k": ml.PyMatrix(d_model, self.d_k, 0.02).to_gpu(),
                "w_v": ml.PyMatrix(d_model, self.d_k, 0.02).to_gpu(),
                "w_out": ml.PyMatrix(self.d_k, d_model, 0.02).to_gpu(),
                "w_gate": ml.PyMatrix(d_model, d_ff, 0.04).to_gpu(),
                "w_up": ml.PyMatrix(d_model, d_ff, 0.04).to_gpu(),
                "w_down": ml.PyMatrix(d_ff, d_model, 0.04).to_gpu()
            }
            self.layers.append(layer)
        self.kv_caches = [ml.KvCache(128, self.d_k) for _ in range(num_layers)]

    def forward(self, x, step):
        out = x
        for idx, layer in enumerate(self.layers):
            norm_x = out.gpu_rms_norm(self.eps)
            q = norm_x.gpu_matmul(layer["w_q"], False)
            k = norm_x.gpu_matmul(layer["w_k"], False)
            v = norm_x.gpu_matmul(layer["w_v"], False)
            q_r = q.gpu_rope()
            k_r = k.gpu_rope()
            self.kv_caches[idx].update(step, k_r, v)
            attn = q_r.gpu_fused_attention(self.kv_caches[idx], step + 1)
            attn_out = attn.gpu_matmul(layer["w_out"], False)
            res = attn_out.gpu_residual_rms_norm(out, self.eps)
            gate = res.gpu_matmul(layer["w_gate"], False)
            up = res.gpu_matmul(layer["w_up"], False)
            swiglu = gate.gpu_swiglu(up)
            mlp = swiglu.gpu_matmul(layer["w_down"], False)
            out = mlp.gpu_add(attn_out)
        return out

    def adapt_weights(self, target_signal, lr=0.0005):
        for layer in self.layers:
            adjustment = layer["w_q"].gpu_scale(target_signal * lr)
            layer["w_q"] = layer["w_q"].gpu_add(adjustment)
            gate_adjustment = layer["w_gate"].gpu_scale(target_signal * lr)
            layer["w_gate"] = layer["w_gate"].gpu_add(gate_adjustment)

def main():
    print("=" * 75)
    print("SOVEREIGN STABLE TRANSFORMER CHARACTER RUNTIME")
    print("=" * 75)
    
    model = SovereignTransformerModel()
    allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz 0123456789.,!?"
    adaptation_state = 0.5
    turn_counter = 0
    
    print("Type your message. The model updates dynamically without freezing.\n" + "-"*50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'exit':
                print("Exiting Sovereign Runtime. Goodbye!")
                break
            
            turn_counter += 1
            input_signal = (sum(ord(c) for c in user_input) % 1000 + random.uniform(-10, 10)) / 1000.0
            model.adapt_weights(input_signal, lr=0.0005)
            
            print("AI: ", end="", flush=True)
            start = time.perf_counter()
            
            adaptation_state = (adaptation_state * 31.5 + turn_counter * 7.1 + random.uniform(0.1, 9.9)) % 100.0 / 100.0
            
            tensor_data = [
                ((i * 37 + ord(user_input[i % len(user_input)]) * 19 + adaptation_state * 1000 + random.uniform(-5, 5)) % 255) / 255.0
                for i in range(256)
            ]
            x = ml.PyMatrix.from_list(1, 256, tensor_data).to_gpu()
            
            generation_length = 36
            for step in range(generation_length):
                x = model.forward(x, step)
                
                cpu_tensor = x.to_cpu()
                data = cpu_tensor.to_list()
                
                tensor_val = sum(data[:64]) * (step + 1.5) + adaptation_state * 500.0 + random.uniform(-2, 2)
                char_index = int(abs(tensor_val * 1331.7)) % len(allowed_chars)
                
                char = allowed_chars[char_index]
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.02)
                
            dur = time.perf_counter() - start
            tps = generation_length / dur if dur > 0 else 0.0
            print(f"\n   [Speed: {tps:.2f} TPS | State: {adaptation_state:.4f}]")
        except KeyboardInterrupt:
            print("\nSession terminated.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break

if __name__ == "__main__":
    main()