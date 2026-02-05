//! MatMul: matrix multiply. Forward a@b; backward grad_a=grad_out@b^T, grad_b=a^T@grad_out.

use super::{Op, OpError, OpId, OpResult};
use crate::tensor::Tensor;

pub struct MatMul;

impl Op for MatMul {
    fn id(&self) -> OpId {
        OpId::MatMul
    }

    fn name(&self) -> &'static str {
        "MatMul"
    }

    fn forward(&self, inputs: &[&Tensor]) -> OpResult<Tensor> {
        if inputs.len() != 2 {
            return Err(OpError("MatMul requires 2 inputs".into()));
        }
        inputs[0]
            .matmul(inputs[1])
            .map_err(|e| OpError(e.to_string()))
    }

    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        _fwd_output: &Tensor,
    ) -> OpResult<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OpError("MatMul backward requires 2 inputs".into()));
        }
        let a = inputs[0];
        let b = inputs[1];
        let b_t = b.transpose().map_err(|e| OpError(e.to_string()))?;
        let a_t = a.transpose().map_err(|e| OpError(e.to_string()))?;
        let grad_a = grad_out.matmul(&b_t).map_err(|e| OpError(e.to_string()))?;
        let grad_b = a_t.matmul(grad_out).map_err(|e| OpError(e.to_string()))?;
        Ok(vec![grad_a, grad_b])
    }
}
