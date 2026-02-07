//! Log: forward log(a); backward grad_out / a.

use super::{Op, OpError, OpId, OpResult};
use crate::tensor::Tensor;

pub struct Log;

impl Op for Log {
    fn id(&self) -> OpId {
        OpId::Log
    }

    fn name(&self) -> &'static str {
        "Log"
    }

    fn forward(&self, inputs: &[&Tensor]) -> OpResult<Tensor> {
        if inputs.len() != 1 {
            return Err(OpError("Log requires 1 input".into()));
        }
        inputs[0].log().map_err(|e| OpError(e.to_string()))
    }

    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        _fwd_output: &Tensor,
    ) -> OpResult<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OpError("Log backward requires 1 input".into()));
        }
        let input = inputs[0];
        let grad = grad_out.div(input).map_err(|e| OpError(e.to_string()))?;
        Ok(vec![grad])
    }
}
