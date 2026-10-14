/// Perceptron
pub struct Perceptron {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub lr: f32,
}

impl Perceptron {

    pub fn new(num_inputs: usize, lr: f32) -> Self {
        Perceptron {
            weights: vec![0.0; num_inputs],
            bias: 0.0,
            lr,
        }
    }

    /// دالة التنبؤ (Forward Pass / Predict)
    pub fn predict(&self, inputs: &[f32]) -> u8 {
        let mut sum = self.bias;
        for i in 0..inputs.len() {
            sum += inputs[i] * self.weights[i];
        }

        if sum >= 0.0 {
            1
        } else {
            0
        }
    }

    pub fn train(&mut self, inputs: &[f32], target: u8) {
        let prediction = self.predict(inputs);
        let error = target as f32 - prediction as f32;

        if error != 0.0 {
            for i in 0..self.weights.len() {
                self.weights[i] += self.lr * error * inputs[i];
            }
            self.bias += self.lr * error;
        }
    }
}