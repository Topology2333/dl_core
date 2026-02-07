//! Softmax along last dimension. Forward: softmax_last_dim; backward: y * (grad_out - sum(grad_out * y)).

use super::{Op, OpError, OpId, OpResult};
use crate::tensor::Tensor;

pub struct Softmax;

impl Op for Softmax {
    fn id(&self) -> OpId {
        OpId::Softmax
    }

    fn name(&self) -> &'static str {
        "Softmax"
    }

    fn forward(&self, inputs: &[&Tensor]) -> OpResult<Tensor> {
        if inputs.len() != 1 {
            return Err(OpError("Softmax requires 1 input".into()));
        }
        inputs[0].softmax_last_dim().map_err(|e| OpError(e.to_string()))
    }

    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        fwd_output: &Tensor,
    ) -> OpResult<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OpError("Softmax backward requires 1 input".into()));
        }
        let grad = fwd_output
            .softmax_backward(grad_out)
            .map_err(|e| OpError(e.to_string()))?;
        Ok(vec![grad])
    }
}
