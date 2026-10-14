pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum).collect()
}

pub struct SoftmaxClassifier {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl SoftmaxClassifier {
    pub fn new(num_inputs: usize, num_classes: usize) -> Self {
        let weights = vec![vec![0.01; num_inputs]; num_classes];
        let biases = vec![0.0; num_classes];
        SoftmaxClassifier { weights, biases }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        let mut logits = vec![0.0; self.biases.len()];
        for i in 0..self.biases.len() {
            let mut sum = self.biases[i];
            for j in 0..inputs.len() {
                sum += self.weights[i][j] * inputs[j];
            }
            logits[i] = sum;
        }
        softmax(&logits)
    }

    pub fn train(&mut self, inputs: &[f64], target_class: usize, lr: f64) {
        let probs = self.forward(inputs);

        for i in 0..self.biases.len() {
            let error = if i == target_class { probs[i] - 1.0 } else { probs[i] };
            
            for j in 0..inputs.len() {
                self.weights[i][j] -= lr * error * inputs[j];
            }
            self.biases[i] -= lr * error;
        }
    }
}