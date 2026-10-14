import math
import random
import ml_core_engine

class DenseLayer:
    def __init__(self, n_inputs, n_outputs):
        scale = math.sqrt(2.0 / n_inputs)
        weights_data = [random.gauss(0.0, scale) for _ in range(n_inputs * n_outputs)]
        
        self.weights = ml_core_engine.PyMatrix.from_list(n_inputs, n_outputs, weights_data)
        self.bias = ml_core_engine.PyMatrix(1, n_outputs, 0.0)
        
        self.inputs = None
        self.output = None
        self.z = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = inputs.gpu_matmul(self.weights)
        self.output = self.z.gpu_relu()
        return self.output

    def backward(self, dvalues, learning_rate):
        relu_grad = self.z.gpu_relu_backward()
        dized = dvalues.multiply(relu_grad)
        
        weights_transpose = self.inputs.transpose()
        dweights = weights_transpose.gpu_matmul(dized)
        
        grad_scaled = dweights.scale(learning_rate)
        self.weights = self.weights.sub(grad_scaled)
        
        return dized.gpu_matmul(self.weights.transpose())

class SimpleMLP:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1]))

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
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.6f}")


if __name__ == "__main__":
    random.seed(42)
    
    X = ml_core_engine.PyMatrix.from_list(1, 2, [0.5, -0.8])
    y = ml_core_engine.PyMatrix.from_list(1, 1, [0.9])
    
    mlp = SimpleMLP([2, 4, 1])
    
    print("--- 🚀 Training Neural Network on GPU via Rust Core ---")
    mlp.train(X, y, epochs=200, learning_rate=0.05)