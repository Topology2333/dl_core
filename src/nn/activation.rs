//! Activation layers: ReLU, Sigmoid (no parameters).

use super::module::Module;
use super::Layer;
use crate::autograd::{Graph, NodeId};
use crate::parameter::Parameter;
use crate::tensor::Tensor;

/// ReLU: max(0, x). No parameters.
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    fn parameters(&self) -> Vec<&Parameter> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![]
    }

    fn forward(&self, x: &Tensor) -> crate::TensorResult<Tensor> {
        x.relu()
    }

    fn forward_graph(
        &self,
        g: &mut Graph,
        x_id: NodeId,
    ) -> crate::GraphResult<(NodeId, Vec<NodeId>)> {
        let out_id = g.relu(x_id)?;
        Ok((out_id, vec![]))
    }
}

impl Layer for ReLU {}

/// Sigmoid: 1/(1+exp(-x)). No parameters.
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sigmoid {
    fn parameters(&self) -> Vec<&Parameter> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![]
    }

    fn forward(&self, x: &Tensor) -> crate::TensorResult<Tensor> {
        x.sigmoid()
    }

    fn forward_graph(
        &self,
        g: &mut Graph,
        x_id: NodeId,
    ) -> crate::GraphResult<(NodeId, Vec<NodeId>)> {
        let out_id = g.sigmoid(x_id)?;
        Ok((out_id, vec![]))
    }
}

impl Layer for Sigmoid {}
