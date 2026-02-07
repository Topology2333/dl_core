//! Simple MLP: Linear -> ReLU -> Linear. For classification use with CE loss.

use super::module::Module;
use super::Layer;
use crate::autograd::{Graph, NodeId};
use crate::parameter::Parameter;
use crate::tensor::Tensor;
use std::sync::Arc;

/// Two-layer MLP: linear1 -> ReLU -> linear2.
pub struct MLP2 {
    pub linear1: super::Linear,
    pub linear2: super::Linear,
}

impl MLP2 {
    pub fn new(
        in_features: usize,
        hidden: usize,
        out_features: usize,
        backend: Arc<dyn crate::backend::Backend>,
    ) -> crate::TensorResult<Self> {
        let linear1 = super::Linear::new(in_features, hidden, backend.clone())?;
        let linear2 = super::Linear::new(hidden, out_features, backend)?;
        Ok(MLP2 { linear1, linear2 })
    }

    pub fn init_xavier(&mut self) -> crate::TensorResult<()> {
        self.linear1.init_xavier()?;
        self.linear2.init_xavier()?;
        Ok(())
    }
}

impl Module for MLP2 {
    fn parameters(&self) -> Vec<&Parameter> {
        self.linear1
            .parameters()
            .into_iter()
            .chain(self.linear2.parameters())
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let p1 = self.linear1.parameters_mut();
        let p2 = self.linear2.parameters_mut();
        p1.into_iter().chain(p2).collect()
    }

    fn forward(&self, x: &Tensor) -> crate::TensorResult<Tensor> {
        let h = self.linear1.forward(x)?;
        let h = h.relu()?;
        self.linear2.forward(&h)
    }

    fn forward_graph(
        &self,
        g: &mut Graph,
        x_id: NodeId,
    ) -> crate::GraphResult<(NodeId, Vec<NodeId>)> {
        let (h_id, mut param_ids) = self.linear1.forward_graph(g, x_id)?;
        let relu_id = g.relu(h_id)?;
        let (out_id, p2) = self.linear2.forward_graph(g, relu_id)?;
        param_ids.extend(p2);
        Ok((out_id, param_ids))
    }
}

impl Layer for MLP2 {}
