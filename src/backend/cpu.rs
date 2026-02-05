//! CPU (scalar) backend: reference implementation. Deterministic, single-threaded.

use crate::backend::{Backend, BackendError, BackendResult};
use crate::shape::Shape;
use crate::tensor::Tensor;
use std::sync::Arc;

/// CPU backend: plain loops, deterministic order.
#[derive(Clone)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> BackendResult<Tensor> {
        let ad = a.shape().dims();
        let bd = b.shape().dims();
        if ad.len() != 2 || bd.len() != 2 {
            return Err(BackendError("matmul requires 2D tensors".into()));
        }
        let (m, k1) = (ad[0], ad[1]);
        let (k2, n) = (bd[0], bd[1]);
        if k1 != k2 {
            return Err(BackendError(format!(
                "matmul dim mismatch: {} != {}",
                k1, k2
            )));
        }
        let adata = a.data();
        let bdata = b.data();
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..k1 {
                    s += adata[i * k1 + k] * bdata[k * n + j];
                }
                out[i * n + j] = s;
            }
        }
        Tensor::from_vec(out, Shape::new(vec![m, n]), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> BackendResult<Tensor> {
        if !a.shape().same_as(b.shape()) {
            return Err(BackendError("add: shape mismatch".into()));
        }
        let n = a.numel();
        let mut out = vec![0.0f32; n];
        let ad = a.data();
        let bd = b.data();
        for i in 0..n {
            out[i] = ad[i] + bd[i];
        }
        Tensor::from_vec(out, a.shape().clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> BackendResult<Tensor> {
        if !a.shape().same_as(b.shape()) {
            return Err(BackendError("mul: shape mismatch".into()));
        }
        let n = a.numel();
        let mut out = vec![0.0f32; n];
        let ad = a.data();
        let bd = b.data();
        for i in 0..n {
            out[i] = ad[i] * bd[i];
        }
        Tensor::from_vec(out, a.shape().clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> BackendResult<Tensor> {
        if !a.shape().same_as(b.shape()) {
            return Err(BackendError("sub: shape mismatch".into()));
        }
        let n = a.numel();
        let mut out = vec![0.0f32; n];
        let ad = a.data();
        let bd = b.data();
        for i in 0..n {
            out[i] = ad[i] - bd[i];
        }
        Tensor::from_vec(out, a.shape().clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn relu(&self, a: &Tensor) -> BackendResult<Tensor> {
        let n = a.numel();
        let mut out = vec![0.0f32; n];
        let ad = a.data();
        for i in 0..n {
            out[i] = if ad[i] > 0.0 { ad[i] } else { 0.0 };
        }
        Tensor::from_vec(out, a.shape().clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn sum(&self, a: &Tensor) -> BackendResult<Tensor> {
        let s: f32 = a.data().iter().sum();
        let out = vec![s];
        Tensor::from_vec(out, Shape::new(vec![1]), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn sum_dim(&self, a: &Tensor, dim: usize) -> BackendResult<Tensor> {
        let dims = a.shape().dims();
        if dim >= dims.len() {
            return Err(BackendError("sum_dim: dim out of range".into()));
        }
        let reduced_len = dims[dim];
        let mut out_dims = dims.to_vec();
        out_dims[dim] = 1;
        let out_numel: usize = out_dims.iter().product();
        let mut out = vec![0.0f32; out_numel];
        let ad = a.data();
        // Strides for input: row-major.
        let in_strides: Vec<usize> = dims
            .iter()
            .rev()
            .scan(1, |s, &d| {
                let r = *s;
                *s *= d;
                Some(r)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        for out_linear in 0..out_numel {
            let mut sum = 0.0;
            for k in 0..reduced_len {
                let mut in_linear = 0;
                let mut rem = out_linear;
                for (d, &out_sz) in out_dims.iter().enumerate() {
                    let coord = if d == dim { k } else { rem % out_sz };
                    in_linear += coord * in_strides[d];
                    if d != dim {
                        rem /= out_sz;
                    }
                }
                sum += ad[in_linear];
            }
            out[out_linear] = sum;
        }
        Tensor::from_vec(out, Shape::new(out_dims), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn from_vec(&self, data: Vec<f32>, shape: Shape) -> BackendResult<Tensor> {
        Tensor::from_vec(data, shape, Arc::new(CpuBackend::new())).map_err(|e| BackendError(e.to_string()))
    }

    fn zeros(&self, shape: &Shape) -> BackendResult<Tensor> {
        let n = shape.numel();
        Tensor::from_vec(vec![0.0; n], shape.clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn ones(&self, shape: &Shape) -> BackendResult<Tensor> {
        let n = shape.numel();
        Tensor::from_vec(vec![1.0; n], shape.clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn sigmoid(&self, a: &Tensor) -> BackendResult<Tensor> {
        let n = a.numel();
        let mut out = vec![0.0f32; n];
        let ad = a.data();
        for i in 0..n {
            out[i] = 1.0 / (1.0 + (-ad[i]).exp());
        }
        Tensor::from_vec(out, a.shape().clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn exp(&self, a: &Tensor) -> BackendResult<Tensor> {
        let n = a.numel();
        let mut out = vec![0.0f32; n];
        let ad = a.data();
        for i in 0..n {
            out[i] = ad[i].exp();
        }
        Tensor::from_vec(out, a.shape().clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn log(&self, a: &Tensor) -> BackendResult<Tensor> {
        let n = a.numel();
        let mut out = vec![0.0f32; n];
        let ad = a.data();
        for i in 0..n {
            out[i] = ad[i].ln();
        }
        Tensor::from_vec(out, a.shape().clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn add_broadcast(&self, a: &Tensor, b: &Tensor) -> BackendResult<Tensor> {
        let ad = a.shape().dims();
        let bd = b.shape().dims();
        if ad.len() != 2 || bd.len() != 1 {
            return Err(BackendError(
                "add_broadcast: expect (matrix, vector) e.g. (N,K) + (K)".into(),
            ));
        }
        let (n, k) = (ad[0], ad[1]);
        if bd[0] != k {
            return Err(BackendError("add_broadcast: last dim must match".into()));
        }
        let adata = a.data();
        let bdata = b.data();
        let mut out = vec![0.0f32; n * k];
        for i in 0..n {
            for j in 0..k {
                out[i * k + j] = adata[i * k + j] + bdata[j];
            }
        }
        Tensor::from_vec(out, Shape::new(vec![n, k]), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn transpose(&self, a: &Tensor) -> BackendResult<Tensor> {
        let d = a.shape().dims();
        if d.len() != 2 {
            return Err(BackendError("transpose: requires 2D tensor".into()));
        }
        let (m, n) = (d[0], d[1]);
        let ad = a.data();
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = ad[i * n + j];
            }
        }
        Tensor::from_vec(out, Shape::new(vec![n, m]), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn scale(&self, a: &Tensor, s: f32) -> BackendResult<Tensor> {
        let n = a.numel();
        let mut out = vec![0.0f32; n];
        let ad = a.data();
        for i in 0..n {
            out[i] = ad[i] * s;
        }
        Tensor::from_vec(out, a.shape().clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn relu_backward(&self, grad_out: &Tensor, input: &Tensor) -> BackendResult<Tensor> {
        if !grad_out.shape().same_as(input.shape()) {
            return Err(BackendError("relu_backward: shape mismatch".into()));
        }
        let n = input.numel();
        let mut out = vec![0.0f32; n];
        let gd = grad_out.data();
        let id = input.data();
        for i in 0..n {
            out[i] = if id[i] > 0.0 { gd[i] } else { 0.0 };
        }
        Tensor::from_vec(out, input.shape().clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }

    fn sigmoid_backward(&self, grad_out: &Tensor, fwd_output: &Tensor) -> BackendResult<Tensor> {
        if !grad_out.shape().same_as(fwd_output.shape()) {
            return Err(BackendError("sigmoid_backward: shape mismatch".into()));
        }
        let n = fwd_output.numel();
        let mut out = vec![0.0f32; n];
        let gd = grad_out.data();
        let fd = fwd_output.data();
        for i in 0..n {
            out[i] = gd[i] * fd[i] * (1.0 - fd[i]);
        }
        Tensor::from_vec(out, fwd_output.shape().clone(), Arc::new(CpuBackend::new()))
            .map_err(|e| BackendError(e.to_string()))
    }
}
