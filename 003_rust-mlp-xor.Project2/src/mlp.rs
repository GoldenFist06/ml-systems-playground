fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(output: f64) -> f64 {
    output * (1.0 - output)
}

pub struct Mlp {
    w_hidden: [[f64; 2]; 2],
    b_hidden: [f64; 2],
    w_output: [f64; 2],
    b_output: f64,
}

impl Mlp {
    pub fn new() -> Self {
        Mlp {
            w_hidden: [[0.5, -0.5], [0.5, 0.5]],
            b_hidden: [-0.2, 0.1],
            w_output: [0.7, -0.8],
            b_output: 0.1,
        }
    }

    pub fn forward(&self, inputs: &[f64; 2]) -> (f64, [f64; 2]) {
        let mut h_out = [0.0; 2];
        for i in 0..2 {
            let z = inputs[0] * self.w_hidden[0][i] + inputs[1] * self.w_hidden[1][i] + self.b_hidden[i];
            h_out[i] = sigmoid(z);
        }

        let z_out = h_out[0] * self.w_output[0] + h_out[1] * self.w_output[1] + self.b_output;
        let final_out = sigmoid(z_out);

        (final_out, h_out)
    }

    pub fn train(&mut self, inputs: &[f64; 2], target: f64, lr: f64) {
        let (out, h_out) = self.forward(inputs);

        let out_error = target - out;
        let out_delta = out_error * sigmoid_derivative(out);

        let mut h_delta = [0.0; 2];
        for i in 0..2 {
            let h_error = out_delta * self.w_output[i];
            h_delta[i] = h_error * sigmoid_derivative(h_out[i]);
        }

        for i in 0..2 {
            self.w_output[i] += lr * out_delta * h_out[i];
        }
        self.b_output += lr * out_delta;

        for i in 0..2 {
            for j in 0..2 {
                self.w_hidden[i][j] += lr * h_delta[j] * inputs[i];
            }
            self.b_hidden[i] += lr * h_delta[i];
        }
    }
}