//! Backend abstraction: device-agnostic interface for tensor operations.
//! All matmul/add/relu etc. go through the Backend trait so implementations
//! can be swapped (CPU scalar, SIMD, GPU) without touching autograd or nn.

use crate::tensor::Tensor;
use crate::Shape;
use thiserror::Error;

#[derive(Error, Debug)]
#[error("backend error: {0}")]
pub struct BackendError(pub String);

pub type BackendResult<T> = Result<T, BackendError>;

/// Device-agnostic backend for tensor operations.
/// Tensor holds an Arc<dyn Backend> and delegates all ops here.
pub trait Backend: Send + Sync {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> BackendResult<Tensor>;
    fn add(&self, a: &Tensor, b: &Tensor) -> BackendResult<Tensor>;
    fn mul(&self, a: &Tensor, b: &Tensor) -> BackendResult<Tensor>;
    fn sub(&self, a: &Tensor, b: &Tensor) -> BackendResult<Tensor>;
    fn relu(&self, a: &Tensor) -> BackendResult<Tensor>;
    fn sum(&self, a: &Tensor) -> BackendResult<Tensor>;
    fn sum_dim(&self, a: &Tensor, dim: usize) -> BackendResult<Tensor>;
    fn from_vec(&self, data: Vec<f32>, shape: Shape) -> BackendResult<Tensor>;
    fn zeros(&self, shape: &Shape) -> BackendResult<Tensor>;
    fn ones(&self, shape: &Shape) -> BackendResult<Tensor>;
    fn sigmoid(&self, a: &Tensor) -> BackendResult<Tensor>;
    fn exp(&self, a: &Tensor) -> BackendResult<Tensor>;
    fn log(&self, a: &Tensor) -> BackendResult<Tensor>;
    fn add_broadcast(&self, a: &Tensor, b: &Tensor) -> BackendResult<Tensor>;
    fn transpose(&self, a: &Tensor) -> BackendResult<Tensor>;
    fn scale(&self, a: &Tensor, s: f32) -> BackendResult<Tensor>;
    /// Element-wise a / b (same shape).
    fn div(&self, a: &Tensor, b: &Tensor) -> BackendResult<Tensor>;
    fn relu_backward(&self, grad_out: &Tensor, input: &Tensor) -> BackendResult<Tensor>;
    fn sigmoid_backward(&self, grad_out: &Tensor, fwd_output: &Tensor) -> BackendResult<Tensor>;
    /// Softmax along last dimension. For 2D [B, C], each row sums to 1.
    fn softmax_last_dim(&self, a: &Tensor) -> BackendResult<Tensor>;
    /// Backward for softmax: grad_in = y * (grad_out - sum(grad_out * y, last_dim)).
    fn softmax_backward(&self, grad_out: &Tensor, fwd_output: &Tensor) -> BackendResult<Tensor>;
}

pub mod cpu;
