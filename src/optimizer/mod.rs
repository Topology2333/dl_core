//! Optimizer: updates parameters using gradients. SGD, Adam, etc.

use crate::parameter::Parameter;
use crate::tensor::Tensor;
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

/// Adam: first-order and second-order moment with bias correction.
/// State: (m, v) per parameter, stored in same order as parameters.
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Per-parameter state: (m, v). Extended on first use or when parameter count grows.
    state: Vec<(Tensor, Tensor)>,
    t: u32,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            state: Vec::new(),
            t: 0,
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &mut [&mut Parameter]) -> OptimizerResult<()> {
        self.t += 1;
        let beta1_t = self.beta1.powi(self.t as i32);
        let beta2_t = self.beta2.powi(self.t as i32);

        while self.state.len() < parameters.len() {
            let p = &parameters[self.state.len()];
            let shape = p.data().shape().clone();
            let backend = p.data().backend();
            let zeros = backend
                .zeros(&shape)
                .map_err(|e| OptimizerError(e.to_string()))?;
            self.state.push((zeros.clone(), zeros));
        }

        for (i, p) in parameters.iter_mut().enumerate() {
            if p.is_frozen() {
                continue;
            }
            let grad = match p.grad() {
                Some(g) => g.clone(),
                None => continue,
            };
            let (m, v) = &mut self.state[i];
            let grad_data = grad.data();
            let m_data = m.data_mut();
            let v_data = v.data_mut();
            let param_data = p.data_mut().data_mut();
            if param_data.len() != grad_data.len() {
                return Err(OptimizerError("param and grad shape mismatch".into()));
            }
            let n = param_data.len();

            for j in 0..n {
                let g = grad_data[j];
                m_data[j] = self.beta1 * m_data[j] + (1.0 - self.beta1) * g;
                v_data[j] = self.beta2 * v_data[j] + (1.0 - self.beta2) * g * g;
            }

            let m_hat = 1.0 / (1.0 - beta1_t);
            let v_hat = 1.0 / (1.0 - beta2_t);

            for j in 0..n {
                let m_j = m_data[j] * m_hat;
                let v_j = v_data[j] * v_hat;
                param_data[j] -= self.lr * m_j / (v_j.sqrt() + self.eps);
            }
        }
        Ok(())
    }
}
