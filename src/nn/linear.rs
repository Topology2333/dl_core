//! Linear: y = x @ W + b. One Parameter for weight, one for bias.

use super::module::Module;
use super::Layer;
use crate::autograd::{Graph, NodeId};
use crate::parameter::Parameter;
use crate::tensor::Tensor;
use std::sync::Arc;

/// Linear layer: output = input @ weight + bias.
pub struct Linear {
    pub weight: Parameter,
    pub bias: Parameter,
}

impl Linear {
    /// in_features: input size (last dim), out_features: output size.
    /// Weights are zeros by default; use [init_xavier] or [crate::xavier_uniform] to init.
    pub fn new(in_features: usize, out_features: usize, backend: Arc<dyn crate::backend::Backend>) -> crate::TensorResult<Self> {
        let weight_shape = crate::shape::Shape::new(vec![in_features, out_features]);
        let bias_shape = crate::shape::Shape::new(vec![out_features]);
        let weight_data = backend.zeros(&weight_shape).map_err(crate::tensor::TensorError::from)?;
        let bias_data = backend.zeros(&bias_shape).map_err(crate::tensor::TensorError::from)?;
        Ok(Linear {
            weight: Parameter::new(weight_data),
            bias: Parameter::new(bias_data),
        })
    }

    /// Initialize weight with Xavier uniform and bias with zeros. Call after [Self::new].
    pub fn init_xavier(&mut self) -> crate::TensorResult<()> {
        let backend = self.weight.data().backend();
        let weight_shape = self.weight.data().shape().clone();
        let w = crate::xavier_uniform(&weight_shape, backend)?;
        *self.weight.data_mut() = w;
        Ok(())
    }

    pub fn named(name: impl AsRef<str>, in_features: usize, out_features: usize, backend: Arc<dyn crate::backend::Backend>) -> crate::TensorResult<Self> {
        let prefix = name.as_ref();
        let mut linear = Self::new(in_features, out_features, backend)?;
        linear.weight.set_name(Some(format!("{}.weight", prefix)));
        linear.bias.set_name(Some(format!("{}.bias", prefix)));
        Ok(linear)
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn forward(&self, x: &Tensor) -> crate::TensorResult<Tensor> {
        let out = x.matmul(self.weight.data())?;
        out.add_broadcast(self.bias.data())
    }

    fn forward_graph(
        &self,
        g: &mut Graph,
        x_id: NodeId,
    ) -> crate::GraphResult<(NodeId, Vec<NodeId>)> {
        let w_id = g.var(self.weight.data().clone());
        let b_id = g.var(self.bias.data().clone());
        let matmul_id = g.matmul(x_id, w_id)?;
        let out_id = g.add_broadcast(matmul_id, b_id)?;
        Ok((out_id, vec![w_id, b_id]))
    }
}

impl Layer for Linear {}
