//! Autograd: computation graph, backward pass, gradient accumulation.
//! Composable (arbitrary nesting), debuggable (inspect data/grad per node), verifiable (numerical grad check).

pub mod graph;
pub mod check;

pub use graph::{Graph, GraphError, GraphResult, Node, NodeId};
