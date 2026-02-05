//! Sub: element-wise subtraction. Forward a-b; backward grad_a=grad_out, grad_b=-grad_out.

use super::{Op, OpError, OpId, OpResult};
use crate::tensor::Tensor;

pub struct Sub;

impl Op for Sub {
    fn id(&self) -> OpId {
        OpId::Sub
    }

    fn name(&self) -> &'static str {
        "Sub"
    }

    fn forward(&self, inputs: &[&Tensor]) -> OpResult<Tensor> {
        if inputs.len() != 2 {
            return Err(OpError("Sub requires 2 inputs".into()));
        }
        inputs[0]
            .sub(inputs[1])
            .map_err(|e| OpError(e.to_string()))
    }

    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        _fwd_output: &Tensor,
    ) -> OpResult<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OpError("Sub backward requires 2 inputs".into()));
        }
        let neg = grad_out.scale(-1.0).map_err(|e| OpError(e.to_string()))?;
        Ok(vec![grad_out.clone(), neg])
    }
}
