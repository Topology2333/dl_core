//! Sum: reduce to scalar. Forward sum(a); backward grad = broadcast grad_out to input shape.

use super::{Op, OpError, OpId, OpResult};
use crate::tensor::Tensor;

pub struct Sum;

impl Op for Sum {
    fn id(&self) -> OpId {
        OpId::Sum
    }

    fn name(&self) -> &'static str {
        "Sum"
    }

    fn forward(&self, inputs: &[&Tensor]) -> OpResult<Tensor> {
        if inputs.len() != 1 {
            return Err(OpError("Sum requires 1 input".into()));
        }
        inputs[0].sum().map_err(|e| OpError(e.to_string()))
    }

    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        _fwd_output: &Tensor,
    ) -> OpResult<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OpError("Sum backward requires 1 input".into()));
        }
        let input = inputs[0];
        if grad_out.numel() != 1 {
            return Err(OpError("Sum backward: grad_out must be scalar".into()));
        }
        let scalar = grad_out.data()[0];
        let ones = input
            .backend()
            .ones(input.shape())
            .map_err(|e| OpError(e.to_string()))?;
        let grad = ones.scale(scalar).map_err(|e| OpError(e.to_string()))?;
        Ok(vec![grad])
    }
}
