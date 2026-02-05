//! dl_core: layered deep learning core with autograd, parameters, and training loop.
//!
//! Layers: storage (Tensor, Shape, Backend) -> autograd (graph, ops) -> nn (Module, Layer)
//! -> train (Trainer, Optimizer, DataLoader).
//!
//! Determinism: use [set_seed] before init/training for reproducible runs.

pub mod autograd;
pub mod backend;
pub mod data;
pub mod init;
pub mod nn;
pub mod ops;
pub mod optimizer;
pub mod parameter;
pub mod runtime;
pub mod shape;
pub mod tensor;
pub mod train;

pub use autograd::{Graph, GraphError, GraphResult, NodeId};
pub use backend::{cpu::CpuBackend, Backend, BackendError, BackendResult};
pub use data::{DataLoader, Dataset, InMemoryDataset};
pub use init::{he_uniform, xavier_uniform};
pub use nn::{Linear, Module, ReLU, Sigmoid, mse, mse_graph};
pub use runtime::{set_seed, with_rng};
pub use ops::{Op, OpId, OpRegistry, OpResult};
pub use optimizer::{Optimizer, OptimizerError, SGD};
pub use parameter::{Parameter, ParameterState};
pub use shape::{Shape, ShapeError};
pub use tensor::{Tensor, TensorError, TensorResult};
pub use train::{Trainer, TrainError, TrainResult, TrainStepResult};
