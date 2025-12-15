use std::vec::Vec;

#[derive(Debug, Clone)]
struct House {
    area: f64,
    bedrooms: f64,
    price: f64,
}

// simple single neuron network
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl Neuron {
    fn new(num_features: usize, learning_rate: f64) -> Self {
        Neuron {
            weights: vec![0.1; num_features],
            bias: 0.0,
            learning_rate,
        }
    }

    fn predict(&self, inputs: &[f64]) -> f64 {
        let weighted_sum: f64 = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum();

        weighted_sum + self.bias
    }

    fn train(&mut self, inputs: &[f64], target: f64) -> f64 {
        let prediction = self.predict(inputs);
        let error = prediction - target;
        let loss = error * error;

        for i in 0..self.weights.len() {
            let gradient_w = 2.0 * error * inputs[i];
            self.weights[i] -= self.learning_rate * gradient_w;
        }

        let gradient_b = 2.0 * error;
        self.bias -= self.learning_rate * gradient_b;

        loss
    }
}

fn main() {
    let houses = vec![
        House {
            area: 140.0,
            bedrooms: 3.0,
            price: 300.0,
        },
        House {
            area: 185.0,
            bedrooms: 4.0,
            price: 400.0,
        },
        House {
            area: 75.0,
            bedrooms: 2.0,
            price: 150.0,
        },
        House {
            area: 230.0,
            bedrooms: 4.0,
            price: 450.0,
        },
    ];

    let max_area = 300.0;
    let max_bedrooms = 5.0;
    let max_price = 500.0;

    let training_data: Vec<(Vec<f64>, f64)> = houses
        .iter()
        .map(|h| {
            (
                vec![h.area / max_area, h.bedrooms / max_bedrooms],
                h.price / max_price,
            )
        })
        .collect();

    let mut neuron = Neuron::new(2, 0.01);

    let epochs = 5000;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (inputs, target) in &training_data {
            total_loss += neuron.train(inputs, *target);
        }

        if epoch % 1000 == 0 {
            println!(
                "Epoch {}: Loss = {:.6}",
                epoch,
                total_loss / training_data.len() as f64
            );
        }
    }

    println!("completed");
    println!("weights: {:.2?}", neuron.weights);
    println!("bias: {:.2}", neuron.bias);

    let new_house = House {
        area: 205.0,
        bedrooms: 3.0,
        price: 0.0,
    };

    let new_input = vec![new_house.area / max_area, new_house.bedrooms / max_bedrooms];
    let normalized_prediction = neuron.predict(&new_input);
    let predicted_price = normalized_prediction * max_price;

    println!(
        "Prediction for house with {:.0} sq meters and {:.0} bedrooms:",
        new_house.area, new_house.bedrooms
    );
    println!("Estimated Price: ${:.2}k", predicted_price);
}
