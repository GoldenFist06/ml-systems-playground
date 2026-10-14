mod perceptron;
use perceptron::Perceptron;

fn main() {
    println!("=== Rust Perceptron - AND Gate Training ===\n");

    let training_data = vec![
        (vec![0.0, 0.0], 0),
        (vec![0.0, 1.0], 0),
        (vec![1.0, 0.0], 0),
        (vec![1.0, 1.0], 1),
    ];

    let mut model = Perceptron::new(2, 0.1);

    let epochs = 10;
    for epoch in 1..=epochs {
        for (inputs, target) in &training_data {
            model.train(inputs, *target);
        }
        println!("Epoch {}/{} completed.", epoch, epochs);
    }

    println!("\n=== Evaluation Results ===");
    for (inputs, target) in &training_data {
        let prediction = model.predict(inputs);
        println!(
            "Input: {:?} | Target: {} | Predicted: {}",
            inputs, target, prediction
        );
    }
}