//! Tensor: pure numerical storage and shape. No grad, no graph (those live in autograd).
//! All ops (matmul, add, relu) are invoked via the Backend trait.

use crate::backend::{Backend, BackendError, BackendResult};
use crate::shape::{Shape, ShapeError};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("backend error: {0}")]
    Backend(#[from] BackendError),
    #[error("shape error: {0}")]
    Shape(#[from] ShapeError),
}

pub type TensorResult<T> = Result<T, TensorError>;

/// Tensor: data + shape + backend reference. No gradient or graph node.
#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
    backend: Arc<dyn Backend>,
}

impl Tensor {
    /// Create a tensor from data and shape using the given backend.
    /// Caller must ensure data.len() == shape.numel().
    pub fn from_vec(data: Vec<f32>, shape: Shape, backend: Arc<dyn Backend>) -> TensorResult<Self> {
        if data.len() != shape.numel() {
            return Err(TensorError::Shape(ShapeError(format!(
                "data len {} != shape numel {}",
                data.len(),
                shape.numel()
            ))));
        }
        Ok(Tensor {
            data,
            shape,
            backend,
        })
    }

    /// Create via backend (backend allocates/copies).
    pub fn from_vec_backend(
        data: Vec<f32>,
        shape: Shape,
        backend: Arc<dyn Backend>,
    ) -> BackendResult<Self> {
        backend.from_vec(data, shape)
    }

    /// Raw data slice (read-only view).
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable data slice (for in-place updates, e.g. optimizer).
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Shape of this tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Backend used by this tensor.
    pub fn backend(&self) -> Arc<dyn Backend> {
        Arc::clone(&self.backend)
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Matrix multiply: self @ rhs. (M,K) @ (K,N) -> (M,N)
    pub fn matmul(&self, rhs: &Tensor) -> TensorResult<Tensor> {
        self.backend.matmul(self, rhs).map_err(TensorError::from)
    }

    /// Element-wise add.
    pub fn add(&self, rhs: &Tensor) -> TensorResult<Tensor> {
        self.backend.add(self, rhs).map_err(TensorError::from)
    }

    /// Element-wise multiply.
    pub fn mul(&self, rhs: &Tensor) -> TensorResult<Tensor> {
        self.backend.mul(self, rhs).map_err(TensorError::from)
    }

    /// Element-wise subtract.
    pub fn sub(&self, rhs: &Tensor) -> TensorResult<Tensor> {
        self.backend.sub(self, rhs).map_err(TensorError::from)
    }

    /// ReLU.
    pub fn relu(&self) -> TensorResult<Tensor> {
        self.backend.relu(self).map_err(TensorError::from)
    }

    /// Sum to scalar.
    pub fn sum(&self) -> TensorResult<Tensor> {
        self.backend.sum(self).map_err(TensorError::from)
    }

    /// Sum along dimension.
    pub fn sum_dim(&self, dim: usize) -> TensorResult<Tensor> {
        self.backend.sum_dim(self, dim).map_err(TensorError::from)
    }

    /// Sigmoid.
    pub fn sigmoid(&self) -> TensorResult<Tensor> {
        self.backend.sigmoid(self).map_err(TensorError::from)
    }

    /// Broadcast add (e.g. bias + matrix).
    pub fn add_broadcast(&self, rhs: &Tensor) -> TensorResult<Tensor> {
        self.backend.add_broadcast(self, rhs).map_err(TensorError::from)
    }

    /// Fill with zeros (in-place). Used for zero_grad.
    pub fn zero_fill(&mut self) {
        self.data.fill(0.0);
    }

    /// Transpose last two dimensions. For 2D (M,N) -> (N,M).
    pub fn transpose(&self) -> TensorResult<Tensor> {
        self.backend.transpose(self).map_err(TensorError::from)
    }

    /// Scale by scalar: self * s.
    pub fn scale(&self, s: f32) -> TensorResult<Tensor> {
        self.backend.scale(self, s).map_err(TensorError::from)
    }

    /// ReLU backward: grad_out * (self > 0).
    pub fn relu_backward(&self, grad_out: &Tensor) -> TensorResult<Tensor> {
        self.backend.relu_backward(grad_out, self).map_err(TensorError::from)
    }

    /// Sigmoid backward: grad_out * fwd_output * (1 - fwd_output). Self is fwd_output.
    pub fn sigmoid_backward(&self, grad_out: &Tensor) -> TensorResult<Tensor> {
        self.backend.sigmoid_backward(grad_out, self).map_err(TensorError::from)
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("data_len", &self.data.len())
            .finish()
    }
}
