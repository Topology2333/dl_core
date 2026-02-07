//! Module: holds parameters, forward returns Tensor. Layer/Model compose Module.

use crate::autograd::{Graph, NodeId};
use crate::parameter::{Parameter, ParameterState};
use crate::tensor::Tensor;
use std::sync::Arc;

/// Module: has parameters and can forward (pure or with graph).
pub trait Module {
    /// All trainable parameters (order must match forward_graph param nodes).
    fn parameters(&self) -> Vec<&Parameter>;

    /// Mutable parameters (for optimizer).
    fn parameters_mut(&mut self) -> Vec<&mut Parameter>;

    /// Forward (inference): no graph, just Tensor in -> Tensor out.
    fn forward(&self, x: &Tensor) -> crate::TensorResult<Tensor>;

    /// Forward and build graph. Caller has already created input node x_id.
    /// Returns (output_node_id, param_node_ids) so caller can copy grads back to parameters.
    fn forward_graph(
        &self,
        g: &mut Graph,
        x_id: NodeId,
    ) -> crate::GraphResult<(NodeId, Vec<NodeId>)>;

    /// Collect all parameter states in order (for save).
    fn state_dict(&self) -> Vec<ParameterState> {
        self.parameters().iter().map(|p| p.to_state()).collect()
    }

    /// Load parameter states in order (for load). State count must match parameter count.
    fn load_state_dict(
        &mut self,
        states: &[ParameterState],
        backend: Arc<dyn crate::backend::Backend>,
    ) -> crate::TensorResult<()> {
        let mut params = self.parameters_mut();
        if params.len() != states.len() {
            return Err(crate::TensorError::Shape(crate::ShapeError(
                format!(
                    "load_state_dict: got {} states, module has {} parameters",
                    states.len(),
                    params.len()
                )
                .into(),
            )));
        }
        for (p, s) in params.iter_mut().zip(states.iter()) {
            p.load_state(s, backend.clone())?;
        }
        Ok(())
    }
}
