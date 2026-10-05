mod mlp;
use mlp::Mlp;

fn main() {
    let mut network = Mlp::new();

    let dataset = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    let epochs = 20000;
    let lr = 0.5;

    for _ in 0..epochs {
        for (inputs, target) in &dataset {
            network.train(inputs, *target, lr);
        }
    }

    for (inputs, target) in &dataset {
        let (out, _) = network.forward(inputs);
        println!("Input: {:?} | Target: {} | Output: {:.4}", inputs, target, out);
    }
}