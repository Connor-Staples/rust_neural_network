use num::{Float, Num, One, Zero};

pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    None,
}

//implementations of the functions
pub fn activation<T: Float>(func: &ActivationFunction, input: T) -> T where T: Float + Default + PartialOrd + Zero{
    match func {
        
        ActivationFunction::ReLU => {
            if input > T::zero() {
                input
            } else {
                T::zero()
            }
        },

        ActivationFunction::Sigmoid => {
            T::one() / (T::one() + (-input).exp())
        },

        ActivationFunction::None => {
            input    
        }

    }
} 

//implementations of the derivitives of the functions
pub fn derivative<T>(func: &ActivationFunction, input: T) -> T where T: Float + Default + PartialOrd + Zero + One {
    match func {

        ActivationFunction::ReLU => {
            if input > T::zero() {
                T::one()
            } else {
                T::zero()
            }
        },

        ActivationFunction::Sigmoid => {
            (T::one() / (T::one() + (-input).exp())) * (T::one() - (T::one() / (T::one() + (-input).exp())))
        },

        ActivationFunction::None => {
            T::zero()    
        }

    }
}