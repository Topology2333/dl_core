//! Computation graph: nodes, dependency recording, topological sort, backward driver.
//! Each node holds: op (if any), input node ids, data (Tensor), grad (Option<Tensor>).

use crate::ops::{OpId, OpRegistry};
use crate::tensor::Tensor;
use std::collections::HashSet;
use thiserror::Error;

#[derive(Error, Debug)]
#[error("graph error: {0}")]
pub struct GraphError(pub String);

pub type GraphResult<T> = Result<T, GraphError>;

/// A single node in the graph: either a leaf (variable) or an op output.
pub struct Node {
    pub op_id: Option<OpId>,
    pub inputs: Vec<NodeId>,
    pub data: Tensor,
    pub grad: Option<Tensor>,
}

/// Node identifier (index into graph's node list).
pub type NodeId = usize;

/// Computation graph: owns all nodes and drives forward/backward.
pub struct Graph {
    nodes: Vec<Node>,
    registry: OpRegistry,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            registry: OpRegistry::new(),
        }
    }

    /// Create a leaf node (variable). Returns NodeId.
    pub fn var(&mut self, data: Tensor) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            op_id: None,
            inputs: vec![],
            data,
            grad: None,
        });
        id
    }

    /// Get reference to node data.
    pub fn data(&self, id: NodeId) -> GraphResult<&Tensor> {
        self.nodes
            .get(id)
            .map(|n| &n.data)
            .ok_or_else(|| GraphError(format!("invalid node id {}", id)))
    }

    /// Get optional gradient (after backward).
    pub fn grad(&self, id: NodeId) -> GraphResult<Option<&Tensor>> {
        self.nodes
            .get(id)
            .map(|n| n.grad.as_ref())
            .ok_or_else(|| GraphError(format!("invalid node id {}", id)))
    }

    /// Get mutable gradient for accumulation.
    fn grad_mut(&mut self, id: NodeId) -> GraphResult<&mut Option<Tensor>> {
        self.nodes
            .get_mut(id)
            .map(|n| &mut n.grad)
            .ok_or_else(|| GraphError(format!("invalid node id {}", id)))
    }

    /// Run backward from loss node. Fills grad for all nodes that contribute to loss.
    pub fn backward(&mut self, loss_id: NodeId) -> GraphResult<()> {
        let order = self.reverse_topo(loss_id)?;
        let loss_data = self.data(loss_id)?;
        let backend = loss_data.backend();
        let one = backend
            .ones(loss_data.shape())
            .map_err(|e| GraphError(e.to_string()))?;
        *self.grad_mut(loss_id)? = Some(one);

        for node_id in order {
            let (op_id, inputs, data) = {
                let n = &self.nodes[node_id];
                let op_id = match n.op_id {
                    Some(o) => o,
                    None => continue,
                };
                (op_id, n.inputs.clone(), n.data.clone())
            };
            let grad_out = self.grad(node_id)?.cloned().ok_or_else(|| {
                GraphError(format!("missing grad at node {}", node_id))
            })?;
            let op = self
                .registry
                .get(op_id)
                .ok_or_else(|| GraphError(format!("unknown op {:?}", op_id)))?;
            let input_tensors: Vec<&Tensor> = inputs
                .iter()
                .map(|&i| self.data(i).unwrap())
                .collect();
            let grads = op
                .backward(&grad_out, &input_tensors, &data)
                .map_err(|e| GraphError(e.0))?;
            for (i, g) in grads.into_iter().enumerate() {
                let in_id = inputs[i];
                self.accumulate_grad(in_id, g)?;
            }
        }
        Ok(())
    }

    fn accumulate_grad(&mut self, node_id: NodeId, g: Tensor) -> GraphResult<()> {
        let grad = self.grad_mut(node_id)?;
        match grad {
            None => *grad = Some(g),
            Some(ref mut existing) => {
                let summed = existing.add(&g).map_err(|e| GraphError(e.to_string()))?;
                *existing = summed;
            }
        }
        Ok(())
    }

    /// Topological order from loss backward: process loss first, then its inputs, etc.
    /// (Reverse post-order of DFS from loss following input edges.)
    fn reverse_topo(&self, loss_id: NodeId) -> GraphResult<Vec<NodeId>> {
        let mut post_order = Vec::new();
        let mut visited = HashSet::new();
        fn dfs(
            id: NodeId,
            nodes: &[Node],
            visited: &mut HashSet<NodeId>,
            post_order: &mut Vec<NodeId>,
        ) {
            if !visited.insert(id) {
                return;
            }
            if let Some(n) = nodes.get(id) {
                for &in_id in &n.inputs {
                    dfs(in_id, nodes, visited, post_order);
                }
            }
            post_order.push(id);
        }
        dfs(loss_id, &self.nodes, &mut visited, &mut post_order);
        post_order.reverse();
        Ok(post_order)
    }

    /// Apply op: forward and create new node. Returns new NodeId.
    pub fn apply(
        &mut self,
        op_id: OpId,
        inputs: &[NodeId],
    ) -> GraphResult<NodeId> {
        let op = self
            .registry
            .get(op_id)
            .ok_or_else(|| GraphError(format!("unknown op {:?}", op_id)))?;
        let input_tensors: Vec<&Tensor> = inputs
            .iter()
            .map(|&i| self.data(i).map_err(|_| GraphError("invalid input id".into())))
            .collect::<GraphResult<Vec<_>>>()?;
        let data = op.forward(&input_tensors).map_err(|e| GraphError(e.0))?;
        let id = self.nodes.len();
        self.nodes.push(Node {
            op_id: Some(op_id),
            inputs: inputs.to_vec(),
            data,
            grad: None,
        });
        Ok(id)
    }

    /// Add: a + b (same shape)
    pub fn add(&mut self, a: NodeId, b: NodeId) -> GraphResult<NodeId> {
        self.apply(OpId::Add, &[a, b])
    }

    /// AddBroadcast: a (e.g. [N,K]) + b (e.g. [K])
    pub fn add_broadcast(&mut self, a: NodeId, b: NodeId) -> GraphResult<NodeId> {
        self.apply(OpId::AddBroadcast, &[a, b])
    }

    /// Sub: a - b
    pub fn sub(&mut self, a: NodeId, b: NodeId) -> GraphResult<NodeId> {
        self.apply(OpId::Sub, &[a, b])
    }

    /// Mul: a * b
    pub fn mul(&mut self, a: NodeId, b: NodeId) -> GraphResult<NodeId> {
        self.apply(OpId::Mul, &[a, b])
    }

    /// MatMul: a @ b
    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> GraphResult<NodeId> {
        self.apply(OpId::MatMul, &[a, b])
    }

    /// ReLU
    pub fn relu(&mut self, a: NodeId) -> GraphResult<NodeId> {
        self.apply(OpId::ReLU, &[a])
    }

    /// Sum to scalar
    pub fn sum(&mut self, a: NodeId) -> GraphResult<NodeId> {
        self.apply(OpId::Sum, &[a])
    }

    /// Sigmoid
    pub fn sigmoid(&mut self, a: NodeId) -> GraphResult<NodeId> {
        self.apply(OpId::Sigmoid, &[a])
    }

    /// Softmax along last dimension
    pub fn softmax(&mut self, a: NodeId) -> GraphResult<NodeId> {
        self.apply(OpId::Softmax, &[a])
    }

    /// Log (natural log)
    pub fn log(&mut self, a: NodeId) -> GraphResult<NodeId> {
        self.apply(OpId::Log, &[a])
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
