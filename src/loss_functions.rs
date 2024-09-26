use math::matrix::Matrix;
use num::{Zero, Float};
pub enum LossFunction {
    MSE,
}

pub fn calculate_loss<T: Float + Zero>(func: &LossFunction, output: &Matrix<T>, expected_output: &Matrix<T>) -> T {

    match func {
        LossFunction::MSE => {
            let mut loss = T::zero();

            for i in 0..output.dimensions.0 {
                let mut temp = output.values[i][0] - expected_output.values[i][0];

                temp = temp * temp;

                loss = loss + temp;
            }

            loss / T::from(output.dimensions.0).unwrap()
        },
    }
}

pub fn loss_derivative<T: Float + Zero + Default>(func: &LossFunction, output: &Matrix<T>, expected_output: &Matrix<T>) -> Matrix<T> {
    match func {
        LossFunction::MSE => {
            let mut loss: Matrix<T> = Matrix::new(output.dimensions.0, 1);
            let n = T::from(output.dimensions.0).unwrap();
            let two = T::from(2).unwrap();

            for i in 0..output.dimensions.0 {
                loss.values[i][0] = (output.values[i][0] - expected_output.values[i][0]) * (n/two);
            }

            loss
        },
    }
}