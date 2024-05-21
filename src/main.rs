mod matrix;
mod neural_network;

fn main() {
    let x: Vec<usize> = vec![3; 1];

    
    let input = matrix::new(3,1);
    
    
    let nn = neural_network::new(3, x, 2);
    let output = nn.forward_propagate(&input);
    output.print();
    
}
