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


    //This will return a vector of the Matrices of the hidden layer node values
    fn forward_propagate_with_intermediate(&self, input: &Matrix) -> Vec<Matrix> {
        if input.dimension != self.input_nodes.dimension {
            panic!("Input does not match the input node dimensions!");
        }

        let mut out: Vec<Matrix> = vec![];
        let layer_amount = self.weight_layers.len();
        let mut intermediate: Matrix = self.weight_layers[0].multiply(input);
        intermediate = intermediate.add(&self.bias_layers[0]);
        intermediate = ReLU(&intermediate);

        //push on the matrix
        out.push(Matrix {
            values: intermediate.values.clone(),
            dimension: intermediate.dimension.clone(),
        }
        );
        //loop through the remaining weight layers
        for i in 1..layer_amount {
            //multiply and add bias
            intermediate = self.weight_layers[i].multiply(&intermediate);

            //case for when its the last weight layer where the output nodes do not have a bias
            if i != layer_amount - 1{
                intermediate = intermediate.add(&self.bias_layers[i]);
            }
            intermediate = ReLU(&intermediate);

            //add the hidden layer values to the vector 
            out.push(Matrix {
                values: intermediate.values.clone(),
                dimension: intermediate.dimension.clone(),
            }
            );
        }
        
        out
    }



    pub fn back_propagate(&mut self, input: &Matrix, expected_output: &Matrix) {
        
        let node_values = self.forward_propagate_with_intermediate(input);
        let mut layers_index = node_values.len() - 1;
        let output_layer = &node_values[layers_index];
        
        let mut errors: Vec<Matrix> = vec![];

        let error = (output_layer.sub(expected_output)).multiply(&output_layer); //dy/dx node error * output node
        errors.push(error); //add output layer errors

        layers_index -= 1;
        let mut run = true;
        while run {
            let weights = self.weight_layers[layers_index + 1].transpose();
            let error = weights.multiply(&errors[errors.len()-1]).hadamard(&node_values[layers_index]);
            errors.push(error);


            
            if layers_index == 0 {
                run = false
            } else {
                layers_index -= 1;
            }
        }
        errors.reverse(); //output error is now last 
        let learning_rate: f32 = 0.001;



        
        /// GRADIENT DESCENT
        //input node weights first
        for wy in 0..self.weight_layers[0].dimension.0 {
            for wx in 0..self.weight_layers[0].dimension.1 {
                self.weight_layers[0].values[wy][wx] -= learning_rate * (input.values[wx][0] * errors[0].values[wy][0]);
            }
        }
        //iterate over rest of the weights
        for l in 1..self.weight_layers.len() {
            for wy in 0..self.weight_layers[l].dimension.0 {
                for wx in 0..self.weight_layers[l].dimension.1 {
                    self.weight_layers[l].values[wy][wx] -= learning_rate * (node_values[l - 1].values[wx][0] * errors[l].values[wy][0]);
                }
            }

        }

        //BIAS 
        // -1 becasue there is no bias for output nodes
        for l in 0..errors.len()-1 {
            for y in 0..errors[l].dimension.0 {
                self.bias_layers[l].values[y][0] -= learning_rate * errors[l].values[y][0];
            }
        }

        

        
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
    //dont know why i looped both dimensions when this only gets applied to vectors 
    for y in 0..m.dimension.0 {
        for x in 0..m.dimension.1 {
            if result.values[y][x] < 0.0 {
                result.values[y][x] = 0.0
            }
        }
    }
   
    result
}

