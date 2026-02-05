//! Parameter: long-lived, updatable, serializable. Distinct from intermediate tensors.

use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Parameter: wraps a Tensor as a trainable parameter. Can be frozen, named, and serialized.
#[derive(Clone)]
pub struct Parameter {
    /// Underlying tensor (data).
    data: Tensor,
    /// Gradient (set after backward from graph).
    grad: Option<Tensor>,
    /// Optional name for grouping / logging.
    name: Option<String>,
    /// If true, optimizer will not update this parameter.
    frozen: bool,
}

impl Parameter {
    pub fn new(data: Tensor) -> Self {
        Parameter {
            data,
            grad: None,
            name: None,
            frozen: false,
        }
    }

    pub fn named(name: impl Into<String>, data: Tensor) -> Self {
        Parameter {
            data,
            grad: None,
            name: Some(name.into()),
            frozen: false,
        }
    }

    /// Reference to the underlying tensor (data).
    pub fn data(&self) -> &Tensor {
        &self.data
    }

    /// Mutable reference to the underlying tensor (for optimizer updates).
    pub fn data_mut(&mut self) -> &mut Tensor {
        &mut self.data
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    pub fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
    }

    /// Gradient (after backward). None before backward or after zero_grad.
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    /// Set gradient (called after graph.backward to copy grad from node).
    pub fn set_grad(&mut self, g: Option<Tensor>) {
        self.grad = g;
    }

    /// Zero out gradient (clear so next backward can accumulate).
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }
}

/// Serializable parameter state (data only, for save/load).
#[derive(Serialize, Deserialize)]
pub struct ParameterState {
    pub name: Option<String>,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Parameter {
    /// Serialize to state (for saving).
    pub fn to_state(&self) -> ParameterState {
        ParameterState {
            name: self.name.clone(),
            shape: self.data.shape().dims().to_vec(),
            data: self.data.data().to_vec(),
        }
    }

    /// Load from state (data only; backend must match).
    pub fn from_state(state: ParameterState, backend: Arc<dyn crate::backend::Backend>) -> Result<Self, crate::tensor::TensorError> {
        let shape = crate::shape::Shape::new(state.shape);
        let data = backend
            .from_vec(state.data, shape)
            .map_err(crate::tensor::TensorError::from)?;
        Ok(Parameter {
            data,
            grad: None,
            name: state.name,
            frozen: false,
        })
    }
}
