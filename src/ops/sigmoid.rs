//! Sigmoid: 1/(1+exp(-x)). Forward sigmoid(a); backward grad * out * (1-out).

use super::{Op, OpError, OpId, OpResult};
use crate::tensor::Tensor;

pub struct Sigmoid;

impl Op for Sigmoid {
    fn id(&self) -> OpId {
        OpId::Sigmoid
    }

    fn name(&self) -> &'static str {
        "Sigmoid"
    }

    fn forward(&self, inputs: &[&Tensor]) -> OpResult<Tensor> {
        if inputs.len() != 1 {
            return Err(OpError("Sigmoid requires 1 input".into()));
        }
        inputs[0].sigmoid().map_err(|e| OpError(e.to_string()))
    }

    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        fwd_output: &Tensor,
    ) -> OpResult<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OpError("Sigmoid backward requires 1 input".into()));
        }
        let grad = fwd_output
            .sigmoid_backward(grad_out)
            .map_err(|e| OpError(e.to_string()))?;
        Ok(vec![grad])
    }
}
