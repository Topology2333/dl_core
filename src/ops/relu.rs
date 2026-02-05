//! ReLU: max(0,x). Forward relu(a); backward grad = grad_out * (a > 0).

use super::{Op, OpError, OpId, OpResult};
use crate::tensor::Tensor;

pub struct ReLU;

impl Op for ReLU {
    fn id(&self) -> OpId {
        OpId::ReLU
    }

    fn name(&self) -> &'static str {
        "ReLU"
    }

    fn forward(&self, inputs: &[&Tensor]) -> OpResult<Tensor> {
        if inputs.len() != 1 {
            return Err(OpError("ReLU requires 1 input".into()));
        }
        inputs[0].relu().map_err(|e| OpError(e.to_string()))
    }

    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        _fwd_output: &Tensor,
    ) -> OpResult<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OpError("ReLU backward requires 1 input".into()));
        }
        let grad = inputs[0]
            .relu_backward(grad_out)
            .map_err(|e| OpError(e.to_string()))?;
        Ok(vec![grad])
    }
}
