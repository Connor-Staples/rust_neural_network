use math::matrix::Matrix;
use num::Float;
use crate::{loss_functions::calculate_loss, optimizer::Optimizer};

pub enum StepFunction {
    SGD,
    
}

pub fn step<T: Float + std::ops::SubAssign + Sync + Send + 'static + Default>(optimizer: &mut Optimizer<T>, errors: &Vec<Matrix<T>>, intermediate: &Vec<Matrix<T>>) {
    let func = &optimizer.step_function;
    match func {

        StepFunction::SGD => {
            for matrix in 0..optimizer.neural_network.weights.len() {
                let shape = optimizer.neural_network.weights[matrix].dimensions;
                

                for y in 0..shape.0 {
                    for x in 0..shape.1 {
                        optimizer.neural_network.weights[matrix].values[y][x] -= (intermediate[matrix].values[x][0] * errors[matrix].values[y][0]) * optimizer.learning_rate;
                    }
                }


            }
        },

    
    }
}