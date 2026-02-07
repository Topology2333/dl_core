//! Training loop: zero_grad -> forward -> loss -> backward -> optimizer step.
//! Training is a first-class citizen: explicit, controllable.

use crate::autograd::Graph;
use crate::nn::{ce_graph, mse_graph, Module};
use crate::optimizer::Optimizer;
use crate::tensor::Tensor;
use thiserror::Error;

#[derive(Error, Debug)]
#[error("train error: {0}")]
pub struct TrainError(pub String);

pub type TrainResult<T> = Result<T, TrainError>;

/// Result of one training step: loss value and optional metrics.
#[derive(Debug)]
pub struct TrainStepResult {
    pub loss: f32,
}

/// Trainer: holds model, optimizer, loss fn; runs step and epoch.
pub struct Trainer<M, O> {
    pub model: M,
    pub optimizer: O,
}

impl<M: Module, O: Optimizer> Trainer<M, O> {
    pub fn new(model: M, optimizer: O) -> Self {
        Trainer { model, optimizer }
    }

    /// One step: zero_grad -> forward (with graph) -> loss -> backward -> optimizer.step.
    /// batch: (inputs, targets). For simplicity we assume single sample or batched inputs.
    /// Returns loss (scalar).
    pub fn step(
        &mut self,
        _backend: std::sync::Arc<dyn crate::backend::Backend>,
        input: &Tensor,
        target: &Tensor,
    ) -> TrainResult<TrainStepResult> {
        let mut g = Graph::new();
        let x_id = g.var(input.clone());
        let (out_id, param_ids) = self
            .model
            .forward_graph(&mut g, x_id)
            .map_err(|e| TrainError(e.to_string()))?;
        let loss_id = mse_graph(&mut g, out_id, target).map_err(|e| TrainError(e.to_string()))?;

        let mut params = self.model.parameters_mut();
        for p in params.iter_mut() {
            p.zero_grad();
        }
        g.backward(loss_id).map_err(|e| TrainError(e.to_string()))?;

        for (p, &node_id) in params.iter_mut().zip(param_ids.iter()) {
            if let Some(grad) = g.grad(node_id).map_err(|e| TrainError(e.to_string()))? {
                p.set_grad(Some(grad.clone()));
            }
        }

        let loss_data = g.data(loss_id).map_err(|e| TrainError(e.to_string()))?;
        let loss_val = loss_data.data()[0];

        self.optimizer
            .step(&mut params)
            .map_err(|e| TrainError(e.to_string()))?;

        Ok(TrainStepResult { loss: loss_val })
    }

    /// One batch step: forward over batch [B, in], loss = mean over batch, backward, optimizer step.
    /// Input shape [B, in_features], target [B, out_features]. Returns scalar loss for this batch.
    pub fn step_batch(
        &mut self,
        _backend: std::sync::Arc<dyn crate::backend::Backend>,
        input: &Tensor,
        target: &Tensor,
    ) -> TrainResult<TrainStepResult> {
        let mut g = Graph::new();
        let x_id = g.var(input.clone());
        let (out_id, param_ids) = self
            .model
            .forward_graph(&mut g, x_id)
            .map_err(|e| TrainError(e.to_string()))?;
        let loss_id = mse_graph(&mut g, out_id, target).map_err(|e| TrainError(e.to_string()))?;

        let mut params = self.model.parameters_mut();
        for p in params.iter_mut() {
            p.zero_grad();
        }
        g.backward(loss_id).map_err(|e| TrainError(e.to_string()))?;

        for (p, &node_id) in params.iter_mut().zip(param_ids.iter()) {
            if let Some(grad) = g.grad(node_id).map_err(|e| TrainError(e.to_string()))? {
                p.set_grad(Some(grad.clone()));
            }
        }

        let loss_data = g.data(loss_id).map_err(|e| TrainError(e.to_string()))?;
        let loss_val = loss_data.data()[0];

        self.optimizer
            .step(&mut params)
            .map_err(|e| TrainError(e.to_string()))?;

        Ok(TrainStepResult { loss: loss_val })
    }

    /// One batch step for classification: same as step_batch but uses cross-entropy loss.
    /// Target must be one-hot [B, num_classes].
    pub fn step_batch_ce(
        &mut self,
        _backend: std::sync::Arc<dyn crate::backend::Backend>,
        input: &Tensor,
        target: &Tensor,
    ) -> TrainResult<TrainStepResult> {
        let mut g = Graph::new();
        let x_id = g.var(input.clone());
        let (out_id, param_ids) = self
            .model
            .forward_graph(&mut g, x_id)
            .map_err(|e| TrainError(e.to_string()))?;
        let loss_id = ce_graph(&mut g, out_id, target).map_err(|e| TrainError(e.to_string()))?;

        let mut params = self.model.parameters_mut();
        for p in params.iter_mut() {
            p.zero_grad();
        }
        g.backward(loss_id).map_err(|e| TrainError(e.to_string()))?;

        for (p, &node_id) in params.iter_mut().zip(param_ids.iter()) {
            if let Some(grad) = g.grad(node_id).map_err(|e| TrainError(e.to_string()))? {
                p.set_grad(Some(grad.clone()));
            }
        }

        let loss_data = g.data(loss_id).map_err(|e| TrainError(e.to_string()))?;
        let loss_val = loss_data.data()[0];

        self.optimizer
            .step(&mut params)
            .map_err(|e| TrainError(e.to_string()))?;

        Ok(TrainStepResult { loss: loss_val })
    }

    /// Run one epoch: iterate dataloader by batch, stack each batch, call step_batch once per batch.
    pub fn run_epoch<D: crate::data::Dataset>(
        &mut self,
        backend: std::sync::Arc<dyn crate::backend::Backend>,
        dataloader: &mut crate::data::DataLoader<D>,
    ) -> TrainResult<(f32, usize)> {
        let mut total_loss = 0.0f32;
        let mut num_batches = 0usize;
        while let Some((inputs, targets)) = dataloader.next_batch() {
            let input_batch = Tensor::stack(&inputs, 0).map_err(|e| TrainError(e.to_string()))?;
            let target_batch =
                Tensor::stack(&targets, 0).map_err(|e| TrainError(e.to_string()))?;
            let r = self.step_batch(backend.clone(), &input_batch, &target_batch)?;
            total_loss += r.loss;
            num_batches += 1;
        }
        let avg = if num_batches > 0 {
            total_loss / num_batches as f32
        } else {
            0.0
        };
        Ok((avg, num_batches))
    }
}
