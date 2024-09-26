use crate::{loss_functions::{LossFunction, loss_derivative}, neural_network::NeuralNetwork, activation_functions::{ActivationFunction, derivative}};
use num::Float;
use crate::step_functions::StepFunction;
use math::matrix::Matrix;


pub struct Optimizer<T: Float> {
    pub neural_network: NeuralNetwork<T>,
    pub learning_rate: T, 
    pub step_function: StepFunction,
    pub loss_function: LossFunction,
    pub training_set: (Matrix<T>, Matrix<T>)
}

impl<T> Optimizer<T> where T: Float + Sync + Send + Default + 'static {
    pub fn backpropagate(&mut self, input: &Matrix<T>, intermediate: &Vec<Matrix<T>>, expected_output: &Matrix<T>) -> Vec<Matrix<T>> {
        let mut intermediate = intermediate.clone();
        intermediate.remove(0);
        
        let actual_output = self.neural_network.forward_propagate(&input);

        let mut errors: Vec<Matrix<T>> = vec![];

        let network_layers = self.neural_network.shape.len();
        let func = &self.neural_network.shape[network_layers - 1].1;

        

        

        {   
            let mut output_error: Matrix<T> = Matrix::new(actual_output.values.len(), 1);
            let output_derivative: Vec<T> = intermediate[intermediate.len()-1].values.iter().map(|x| derivative(func, x[0])).collect();
            let cost_derivative: Matrix<T> = loss_derivative(&self.loss_function, &actual_output, expected_output);

            for i in 0..output_derivative.len() {
                output_error.values[i][0] = output_derivative[i] * cost_derivative.values[i][0];
            }

            errors.push(output_error);
        }
        
        // -2 because we just did the output and there is no input error
        for i in (1..self.neural_network.shape.len()-1).rev() {
            
            let func = &self.neural_network.shape[i].1;
            let mut layer_error: Matrix<T> = Matrix::new(self.neural_network.shape[i].0, 1);
            
            let node_derivative: Vec<T> = intermediate[i - 1].values.iter().map(|x| derivative(func, x[0])).collect();

            let cost_derivative: Matrix<T> = (&self.neural_network.weights[i].transpose() * &errors.last().unwrap()).unwrap();
            
            for i in 0..node_derivative.len() {
                layer_error.values[i][0] = node_derivative[i] * cost_derivative.values[i][0];
            }

            errors.push(layer_error);
        }
        
        errors.reverse();
        
        errors

        
    }

   
    
}

impl<T: Float> Optimizer<T> {
    pub fn detatch(self) -> NeuralNetwork<T> {
        self.neural_network
    }

    pub fn new(nn: NeuralNetwork<T>, learning_rate: T, loss_function: LossFunction, step_function: StepFunction, training_set: (Matrix<T>, Matrix<T>)) -> Optimizer<T> {
        Optimizer {
            neural_network: nn,
            learning_rate,
            loss_function,
            step_function,
            training_set,
        
        }
    }
}