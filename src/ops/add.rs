//! Add: element-wise addition. Forward a+b; backward grad_a=grad_out, grad_b=grad_out.

use super::{Op, OpError, OpId, OpResult};
use crate::tensor::Tensor;

pub struct Add;

impl Op for Add {
    fn id(&self) -> OpId {
        OpId::Add
    }

    fn name(&self) -> &'static str {
        "Add"
    }

    fn forward(&self, inputs: &[&Tensor]) -> OpResult<Tensor> {
        if inputs.len() != 2 {
            return Err(OpError("Add requires 2 inputs".into()));
        }
        inputs[0]
            .add(inputs[1])
            .map_err(|e| OpError(e.to_string()))
    }

    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        _fwd_output: &Tensor,
    ) -> OpResult<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OpError("Add backward requires 2 inputs".into()));
        }
        if !grad_out.shape().same_as(inputs[0].shape()) || !grad_out.shape().same_as(inputs[1].shape()) {
            return Err(OpError("Add backward: same-shape only".into()));
        }
        Ok(vec![grad_out.clone(), grad_out.clone()])
    }
}
