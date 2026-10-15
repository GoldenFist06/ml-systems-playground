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
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.6f}")