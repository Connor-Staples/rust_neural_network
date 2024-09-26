

use math::matrix::Matrix;
use num::{Float, One};
use crate::activation_functions::{ActivationFunction, activation};
use rand::{distributions::Standard, prelude::*};

pub struct NeuralNetwork<T> where T: Float {
    pub shape: Vec<(usize, ActivationFunction)>,
    pub weights: Vec<Matrix<T>>,
    pub bias: Vec<Matrix<T>>,
}

impl<T> NeuralNetwork<T> where T: Float + Copy + Default {
    pub fn new(shape: Vec<(usize, ActivationFunction)> ) -> NeuralNetwork<T> {

        //there will always be len-1 weight matricies
        let mut weights: Vec<Matrix<T>> = vec![];
        let mut bias: Vec<Matrix<T>> = vec![];
        for i in 0..shape.len()-1 {
            let weight_mat: Matrix<T> = Matrix::new(shape[i + 1].0 , shape[i].0);
            let bias_mat: Matrix<T> = Matrix::new(shape[i + 1].0, 1);
            weights.push(weight_mat);
            bias.push(bias_mat);
        }

        NeuralNetwork {
            shape,
            weights,
            bias,
        }
    }

    

}
impl<T> NeuralNetwork<T> where T: Float + Copy + Default + Sync + Send + 'static + One {
    pub fn forward_propagate(&self, input: &Matrix<T>) -> Matrix<T> {

        let mut inter = input.clone();
        //println!("there are {} weight matricies", self.weights.len());
        for i in 0..self.weights.len() {
            //multiply weight matrix
            //println!("shape of weights {} x {} : shape of inter {} x 1 : iteration {}",self.weights[i].dimensions.0,self.weights[i].dimensions.1, inter.dimensions.0, i);
            inter = (&self.weights[i] * &inter).unwrap();
            //add bias
            inter = (&inter + &self.bias[i]).unwrap();
            
            //apply activation
            let func = &self.shape[i+1].1;
            for j in 0..inter.dimensions.0 {
                inter.values[j][0] = activation(func, inter.values[j][0]);
            }
        }

        inter
    }

    //calculates and returns layer values before getting put through the activation function
    pub fn forward_propagate_with_intermediate(&self, input: &Matrix<T>) -> Vec<Matrix<T>> {
        let mut inter = input.clone();
        let mut values: Vec<Matrix<T>> = vec![input.clone()];
        for i in 0..self.weights.len() {
            //multiply weight matrix
            inter = (&self.weights[i] * &inter).unwrap();
            //add bias
            inter = (&inter + &self.bias[i]).unwrap();

            values.push(inter.clone());
            
            //apply activation
            let func = &self.shape[i+1].1;
            for j in 0..inter.dimensions.0 {
                inter.values[j][0] = activation(func, inter.values[j][0]);
            }
        }

        values
    }

    //the implementation may potentially be wrong given limited knowledge on distributions
    pub fn he_init(&mut self) where Standard: Distribution<T> {
        let input_nodes = T::from((self.shape[0].0)).unwrap();
        let mut rng = rand::thread_rng();
        for mat in 0..self.weights.len() {

            for y in 0..self.weights[mat].values.len() {
                for x in 0..self.weights[mat].values[0].len()  {
                    self.weights[mat].values[y][x] = rng.gen() * ((T::one() + T::one()).sqrt()/input_nodes);
                }
            }
        }
    }
}