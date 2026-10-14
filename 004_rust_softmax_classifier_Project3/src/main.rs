mod classifier;
use classifier::SoftmaxClassifier;

fn main() {
    let mut classifier = SoftmaxClassifier::new(2, 3);

    let dataset = vec![
        (vec![1.0, 0.0], 0),
        (vec![0.0, 1.0], 1),
        (vec![0.5, 0.5], 2),
    ];

    let epochs = 5000;
    let lr = 0.1;

    for _ in 0..epochs {
        for (inputs, target) in &dataset {
            classifier.train(inputs, *target, lr);
        }
    }

    for (inputs, target) in &dataset {
        let probs = classifier.forward(inputs);
        println!("Input: {:?} | Target: {} | Probs: {:?}", inputs, target, probs);
    }
}