//! Operators as first-class objects: Op trait, registry, forward/backward.
//! Each op (Add, MatMul, ReLU, ...) is an independent entity; adding a new op
//! = implement trait + register, no changes to engine logic.

use crate::tensor::Tensor;
use std::sync::Arc;
use thiserror::Error;

pub mod add;
pub mod add_broadcast;
pub mod sub;
pub mod matmul;
pub mod relu;
pub mod mul;
pub mod sigmoid;
pub mod sum;
pub mod softmax;
pub mod log;

#[derive(Error, Debug)]
#[error("op error: {0}")]
pub struct OpError(pub String);

pub type OpResult<T> = Result<T, OpError>;

/// Unique identifier for an operator type (used in graph to dispatch backward).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpId {
    Add,
    AddBroadcast,
    Sub,
    Mul,
    MatMul,
    ReLU,
    Sigmoid,
    Sum,
    Softmax,
    Log,
}

/// Unified operator trait: forward, backward, and shape constraints.
/// Engine records (OpId, inputs, output) on forward; on backward it looks up
/// the op and calls backward(grad_out, inputs, fwd_output).
pub trait Op: Send + Sync {
    /// Operator id for registration and backward dispatch.
    fn id(&self) -> OpId;

    /// Forward: compute output from inputs.
    fn forward(&self, inputs: &[&Tensor]) -> OpResult<Tensor>;

    /// Backward: given gradient w.r.t. output and saved inputs/output,
    /// return gradients w.r.t. each input (same order as inputs).
    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        fwd_output: &Tensor,
    ) -> OpResult<Vec<Tensor>>;

    /// Name for debugging.
    fn name(&self) -> &'static str {
        "Op"
    }
}

/// Registry: map OpId -> Box<dyn Op>. Engine uses this to run backward.
pub struct OpRegistry {
    ops: std::collections::HashMap<OpId, Arc<dyn Op>>,
}

impl OpRegistry {
    pub fn new() -> Self {
        let mut reg = OpRegistry {
            ops: std::collections::HashMap::new(),
        };
        reg.register(Arc::new(add::Add));
        reg.register(Arc::new(add_broadcast::AddBroadcast));
        reg.register(Arc::new(sub::Sub));
        reg.register(Arc::new(mul::Mul));
        reg.register(Arc::new(matmul::MatMul));
        reg.register(Arc::new(relu::ReLU));
        reg.register(Arc::new(sigmoid::Sigmoid));
        reg.register(Arc::new(sum::Sum));
        reg.register(Arc::new(softmax::Softmax));
        reg.register(Arc::new(log::Log));
        reg
    }

    pub fn register(&mut self, op: Arc<dyn Op>) {
        self.ops.insert(op.id(), op);
    }

    pub fn get(&self, id: OpId) -> Option<Arc<dyn Op>> {
        self.ops.get(&id).cloned()
    }
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self::new()
    }
}
