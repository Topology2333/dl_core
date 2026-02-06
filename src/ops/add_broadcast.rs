//! AddBroadcast: a (e.g. [N,K]) + b (e.g. [K]) with broadcast. Backward: grad_a = grad_out, grad_b = sum(grad_out, dim=0) reshaped to [K].

use super::{Op, OpError, OpId, OpResult};
use crate::shape::Shape;
use crate::tensor::Tensor;

pub struct AddBroadcast;

impl Op for AddBroadcast {
    fn id(&self) -> OpId {
        OpId::AddBroadcast
    }

    fn name(&self) -> &'static str {
        "AddBroadcast"
    }

    fn forward(&self, inputs: &[&Tensor]) -> OpResult<Tensor> {
        if inputs.len() != 2 {
            return Err(OpError("AddBroadcast requires 2 inputs".into()));
        }
        inputs[0]
            .add_broadcast(inputs[1])
            .map_err(|e| OpError(e.to_string()))
    }

    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        _fwd_output: &Tensor,
    ) -> OpResult<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OpError("AddBroadcast backward requires 2 inputs".into()));
        }
        let grad_a = grad_out.clone();
        let summed = grad_out.sum_dim(0).map_err(|e| OpError(e.to_string()))?;
        let k = inputs[1].numel();
        let grad_b = Tensor::from_vec(
            summed.data().to_vec(),
            Shape::new(vec![k]),
            summed.backend(),
        )
        .map_err(|e| OpError(e.to_string()))?;
        Ok(vec![grad_a, grad_b])
    }
}
