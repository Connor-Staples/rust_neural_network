use std::process::Output;


use loss_functions::calculate_loss;
use math::matrix::Matrix;
use neural_network::NeuralNetwork;
use optimizer::Optimizer;
use num::Float;
use num::Zero;
use step_functions::step;

mod neural_network;
mod activation_functions;
mod optimizer;
mod loss_functions;
mod step_functions;

use crate::activation_functions::ActivationFunction::*;
use crate::step_functions::StepFunction;
fn main() {
    let shape = vec![(3, None), (5,ReLU), (5,ReLU), (2,ReLU)];
    let mut nn: NeuralNetwork<f64> = NeuralNetwork::new(shape);
    nn.he_init();

    let input = Matrix::from_vec(&vec![vec![1.0], vec![2.0], vec![3.0]] ).unwrap();
    let output: Matrix<f64> = Matrix::from_vec(&vec![vec![3.0], vec![3.0]]).unwrap();

    input.print();
    let x = nn.forward_propagate(&input);

    println!("Valid");

    let mut optimizer = Optimizer::new(nn, 0.02, loss_functions::LossFunction::MSE, step_functions::StepFunction::SGD, (input.clone(), output.clone()));
    

    for i in 0..3 {
        let costb = calculate_loss(&optimizer.loss_function, &optimizer.neural_network.forward_propagate(&input), &output);
        
        println!("Cost before iteration {} is {}", i, costb);
        let intermediate = optimizer.neural_network.forward_propagate_with_intermediate(&input);
        
        
        let errors = optimizer.backpropagate(&input, &intermediate, &output);
       
        step(&mut optimizer, &errors, &intermediate);
       

        let costa = calculate_loss(&optimizer.loss_function, &optimizer.neural_network.forward_propagate(&input), &output);
        
        println!("Cost after iteration {} is {} : a {}% reduction", i, costa,100.0 - ((costa / costb) * 100.0));
        
    }

    let nn = optimizer.detatch();

    let x = nn.forward_propagate(&Matrix::from_vec(&vec![vec![2.0], vec![4.0], vec![6.0]]).unwrap());
    
    x.print();

   

    
}
