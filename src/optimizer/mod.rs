//! Optimizer: updates parameters using gradients. SGD, Adam, etc.

use crate::parameter::Parameter;
use thiserror::Error;

#[derive(Error, Debug)]
#[error("optimizer error: {0}")]
pub struct OptimizerError(pub String);

pub type OptimizerResult<T> = Result<T, OptimizerError>;

/// Optimizer trait: step(parameters) updates parameters using their gradients.
pub trait Optimizer {
    /// Perform one update step: param -= lr * grad (or equivalent).
    fn step(&mut self, parameters: &mut [&mut Parameter]) -> OptimizerResult<()>;
}

/// SGD: param = param - lr * grad.
pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        SGD { lr }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [&mut Parameter]) -> OptimizerResult<()> {
        for p in parameters.iter_mut() {
            if p.is_frozen() {
                continue;
            }
            let grad = match p.grad() {
                Some(g) => g.clone(),
                None => continue,
            };
            let data = p.data_mut();
            let grad_data = grad.data();
            if data.data().len() != grad_data.len() {
                return Err(OptimizerError("param and grad shape mismatch".into()));
            }
            let d = data.data_mut();
            for i in 0..d.len() {
                d[i] -= self.lr * grad_data[i];
            }
        }
        Ok(())
    }
}
