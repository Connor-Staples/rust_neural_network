use std::process::Output;

mod matrix;
mod neural_network;

fn main() {
    let x: Vec<usize> = vec![3; 1];


    
    let mut input = matrix::new(1,1);
    
    let mut output = matrix::new(1,1);
    
    input.values[0][0] = 2.0;
    output.values[0][0] = 4.0;
    let mut nn = neural_network::new(1, x, 1);
    nn.back_propagate(&input, &output);
    //test to see if it converges on a simple example...
    for i in 0..100 {
        nn.forward_propagate(&input).print();
        nn.back_propagate(&input, &output);
    }
    
   
}
