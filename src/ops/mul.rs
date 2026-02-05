//! Mul: element-wise multiplication. Forward a*b; backward grad_a=grad_out*b, grad_b=grad_out*a.

use super::{Op, OpError, OpId, OpResult};
use crate::tensor::Tensor;

pub struct Mul;

impl Op for Mul {
    fn id(&self) -> OpId {
        OpId::Mul
    }

    fn name(&self) -> &'static str {
        "Mul"
    }

    fn forward(&self, inputs: &[&Tensor]) -> OpResult<Tensor> {
        if inputs.len() != 2 {
            return Err(OpError("Mul requires 2 inputs".into()));
        }
        inputs[0]
            .mul(inputs[1])
            .map_err(|e| OpError(e.to_string()))
    }

    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        _fwd_output: &Tensor,
    ) -> OpResult<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OpError("Mul backward requires 2 inputs".into()));
        }
        let grad_a = grad_out.mul(inputs[1]).map_err(|e| OpError(e.to_string()))?;
        let grad_b = grad_out.mul(inputs[0]).map_err(|e| OpError(e.to_string()))?;
        Ok(vec![grad_a, grad_b])
    }
}
