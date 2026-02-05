//! Module: holds parameters, forward returns Tensor. Layer/Model compose Module.

use crate::autograd::{Graph, NodeId};
use crate::parameter::Parameter;
use crate::tensor::Tensor;

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
}
