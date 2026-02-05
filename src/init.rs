//! Weight initialization: Xavier, He. Pure functions; deterministic for fixed seed.

use crate::runtime::with_rng;
use crate::shape::Shape;
use crate::tensor::Tensor;
use rand::Rng;
use std::sync::Arc;

/// Xavier (Glorot) uniform: scale = sqrt(6 / (fan_in + fan_out)).
/// For 2D weight [fan_in, fan_out], fills with Uniform(-scale, scale).
pub fn xavier_uniform(shape: &Shape, backend: Arc<dyn crate::backend::Backend>) -> crate::TensorResult<Tensor> {
    let dims = shape.dims();
    if dims.len() < 2 {
        return backend
            .zeros(shape)
            .map_err(crate::tensor::TensorError::from);
    }
    let fan_in = dims[0];
    let fan_out = dims[1];
    let scale = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
    let n = shape.numel();
    let data: Vec<f32> = with_rng(|rng| {
        (0..n)
            .map(|_| rng.gen_range(-scale..=scale))
            .collect()
    });
    Tensor::from_vec(data, shape.clone(), backend)
}

/// He (Kaiming) uniform: scale = sqrt(6 / fan_in). For ReLU.
pub fn he_uniform(shape: &Shape, backend: Arc<dyn crate::backend::Backend>) -> crate::TensorResult<Tensor> {
    let dims = shape.dims();
    if dims.is_empty() {
        return backend.zeros(shape).map_err(crate::tensor::TensorError::from);
    }
    let fan_in = dims[0];
    let scale = (6.0f32 / fan_in as f32).sqrt();
    let n = shape.numel();
    let data: Vec<f32> = with_rng(|rng| {
        (0..n)
            .map(|_| rng.gen_range(-scale..=scale))
            .collect()
    });
    Tensor::from_vec(data, shape.clone(), backend)
}
