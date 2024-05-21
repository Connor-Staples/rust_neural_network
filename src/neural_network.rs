use std::vec;

use crate::matrix;
use crate::matrix::Matrix;


pub struct Neural_Network {
    input_nodes: Matrix,
    weight_layers: Vec<Matrix>,
    bias_layers: Vec<Matrix>,
    output_nodes: Matrix,
    
}

impl Neural_Network {
    pub fn overview(&self) {
        println!("This network has {} Input Nodes", self.input_nodes.dimension.0);
        println!("This network has {} Weight Layers", self.weight_layers.len());
        println!("This network has {} Output Nodes", self.output_nodes.dimension.0);
    }

    pub fn forward_propagate(&self, input: &Matrix) -> Matrix {
        if input.dimension != self.input_nodes.dimension {
            panic!("Input does not match the input node dimensions!");
        }
        let layer_amount = self.weight_layers.len();
        let mut intermediate: Matrix = self.weight_layers[0].multiply(input);
        intermediate = intermediate.add(&self.bias_layers[0]);
        intermediate = ReLU(&intermediate);
        //does the matrix multiplication for all the other weights till output
        for i in 1..layer_amount {
            intermediate = self.weight_layers[i].multiply(&intermediate);

            //edge case when at the output nodes
            if i != layer_amount - 1{
                intermediate = intermediate.add(&self.bias_layers[i]);
            }
            intermediate = ReLU(&intermediate);
        }
        intermediate
    }
}

pub fn new(input_nodes: usize, hidden_layers: Vec<usize>, output_nodes: usize) -> Neural_Network {
    let mut weights: Vec<Matrix> = vec![];
    let temp_matrix = matrix::new(hidden_layers[0], input_nodes);
    weights.push(temp_matrix);
    
    //make weights for input layer to first hidden

    for i in 1..hidden_layers.len() { //goes through each value in hidden layers by index
        let temp_matrix = matrix::new(hidden_layers[i-1], hidden_layers[i]);
        weights.push(temp_matrix);
    }

    let temp_matrix = matrix::new(output_nodes, hidden_layers[hidden_layers.len()-1]);
    weights.push(temp_matrix);
    //bias has the same shape as hidden layer
    let mut bias: Vec<Matrix> = vec![];
    for l in hidden_layers {
        bias.push(matrix::new(l, 1));
    }

    let nn = Neural_Network {
        input_nodes: matrix::new(input_nodes, 1),
        weight_layers: weights,
        bias_layers: bias,
        output_nodes: matrix::new(output_nodes, 1),
        
    };
    nn
}   

pub fn ReLU(m : &Matrix) -> Matrix {

    let mut result = Matrix {
        values: m.values.clone(),
        dimension: m.dimension.clone(),
    };
    for y in 0..m.dimension.0 {
        for x in 0..m.dimension.1 {
            if result.values[y][x] < 0.0 {
                result.values[y][x] = 0.0
            }
        }
    }
   
    result
}

