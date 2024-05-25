
use rand::Rng;
pub struct Matrix {
    pub values: Vec<Vec<f32>>,
    pub dimension: (usize, usize),
}

impl Matrix {
    pub fn transpose(&self) -> Matrix {
    let mut vtemp: Vec<f32> = Vec::new();
    let mut mtemp: Vec<Vec<f32>> = Vec::new();
    let mut i = 0;
    //goes through values and makes new vector with transposed values
    while i < self.values[0].len() {
        let mut j = 0;
        while j < self.values.len() {
            vtemp.push(self.values[j][i]);
            j = j + 1;
        }
        
        mtemp.push(vtemp.clone());
        vtemp.clear();
        i = i + 1;
    }
    let m = Matrix {
        values: mtemp,
        dimension: (self.dimension.1, self.dimension.0),
    };
    m
    }
    pub fn sub(&self, m: &Matrix) -> Matrix {
        if (self.dimension.1 != m.dimension.1) {
            panic!("Matrix dimensions cannot be subtracted together");
        }

        let mut result = Matrix {
            values: self.values.clone(),
            dimension: self.dimension,
        };

        for y in 0..self.dimension.0 {
            for x in 0..self.dimension.1 {
                result.values[y][x] -= m.values[y][x];
            }
        }
        result
    }
    pub fn add(&self, m : &Matrix) -> Matrix {
        if (self.dimension.1 != m.dimension.1) {
            panic!("Matrix dimensions cannot be add together");
        }

        let mut result = Matrix {
            values: self.values.clone(),
            dimension: self.dimension,
        };

        for y in 0..self.dimension.0 {
            for x in 0..self.dimension.1 {
                result.values[y][x] += m.values[y][x];
            }
        }
        result
    }
    //will multiply matricies
    pub fn multiply(&self, m : &Matrix) -> Matrix {
        //Mathematically check if they can be multiplied
        if self.dimension.1 != m.dimension.0 {
            panic!("Matrix dimensions cannot be multiplied together");
        }
        let mut multiplied_matrix = new(self.dimension.0, m.dimension.1); //create matrix with the dimensions required
        
        let m_transpose = m.transpose(); // code is cleaner to transpose the second matrix and dot prod the rows
        
        //dot products all the rows
        for i in 0..self.dimension.0 {
            for j in 0..m_transpose.dimension.0 {
                multiplied_matrix.values[i][j] = dot_product(&self.values[i], &m_transpose.values[j]);
            }
            
        }

        multiplied_matrix
    }
    pub fn hadamard(&self, m: &Matrix) -> Matrix {
        if (self.dimension.1 != m.dimension.1) {
            panic!("Matrix dimensions cannot be operated together");
        }
        let mut new_matrix: Matrix = new(self.dimension.0, self.dimension.1);
        for y in 0..self.dimension.0 {
            for x in 0..self.dimension.1 {
                new_matrix.values[y][x] = self.values[y][x] * m.values[y][x];
            }
        }
        new_matrix
    }


    //prints the matrix out in a nice orderly fashion
    pub fn print(&self) {
        for y in 0..self.dimension.0 {
            print!("[");
            for x in 0..self.dimension.1 {
                print!("{} ", self.values[y][x]);
            }
            print!("]");
            println!("");
        }
    }
}

//generates a new matrix a random value between 0-1
pub fn new(y_length: usize, x_length: usize) -> Matrix {
    let v: Vec<Vec<f32>> = vec![vec![rand::thread_rng().gen::<f32>(); x_length]; y_length];
    let m = Matrix {
        values: v,
        dimension: (y_length, x_length),
    };
    m
}

fn dot_product(v1: &Vec<f32>, v2: &Vec<f32>) -> f32 {
    if v1.len() != v2.len() {
        panic!("Vector Lengths do not match in dot_product");
    }
    
    let mut i = 0;
    let mut c:f32 = 0.0;
    while i < v1.len() {
        c = c + v1[i] * v2[i];
        i = i + 1;
    }
    c
}