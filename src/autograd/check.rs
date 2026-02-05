//! Numerical gradient check: finite difference vs autograd for verification.

use crate::tensor::Tensor;

/// Epsilon for central difference: (f(x+eps) - f(x-eps)) / (2*eps).
pub const DEFAULT_EPS: f32 = 1e-4;

/// Compute numerical gradient of scalar function f w.r.t. tensor x via central difference.
/// f: (tensor) -> scalar. Returns gradient same shape as x.
pub fn numerical_grad(
    x: &Tensor,
    f: impl Fn(&Tensor) -> f32,
    eps: f32,
) -> Vec<f32> {
    let n = x.numel();
    let mut grad = vec![0.0f32; n];
    let data = x.data();
    let backend = x.backend();
    let shape = x.shape().clone();
    for i in 0..n {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        let t_plus = Tensor::from_vec(plus, shape.clone(), backend.clone()).unwrap();
        let t_minus = Tensor::from_vec(minus, shape.clone(), backend.clone()).unwrap();
        grad[i] = (f(&t_plus) - f(&t_minus)) / (2.0 * eps);
    }
    grad
}

/// Check gradients: compare autograd grad at each input node with numerical gradient.
/// build_loss: (g, input_ids) -> loss_node_id. Builds graph and returns loss id.
/// input_tensors: initial tensor for each input (same order as input_ids).
/// Returns Ok(()) if all gradients match within rtol/atol.
pub fn check_gradients(
    build_loss: &impl Fn(&mut crate::autograd::Graph, &[crate::autograd::NodeId]) -> crate::GraphResult<crate::autograd::NodeId>,
    input_tensors: &[Tensor],
    eps: f32,
    rtol: f32,
    atol: f32,
) -> Result<(), String> {
    use crate::autograd::{Graph, NodeId};
    let mut g = Graph::new();
    let input_ids: Vec<NodeId> = input_tensors
        .iter()
        .map(|t| g.var(t.clone()))
        .collect();
    let loss_id = build_loss(&mut g, &input_ids).map_err(|e| e.to_string())?;
    g.backward(loss_id).map_err(|e| e.to_string())?;

    for (idx, (input_id, input_tensor)) in input_ids.iter().zip(input_tensors.iter()).enumerate() {
        let autograd_grad = g
            .grad(*input_id)
            .map_err(|e| e.to_string())?
            .and_then(|t| Some(t.data().to_vec()))
            .ok_or_else(|| format!("missing grad at input {}", idx))?;

        let idx_capture = idx;
        let num_grad = numerical_grad(input_tensor, move |perturbed: &Tensor| {
            let mut g2 = Graph::new();
            let ids: Vec<NodeId> = input_tensors
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    if i == idx_capture {
                        g2.var(perturbed.clone())
                    } else {
                        g2.var(t.clone())
                    }
                })
                .collect();
            let lid = build_loss(&mut g2, &ids).unwrap();
            g2.data(lid).unwrap().data()[0]
        }, eps);

        if autograd_grad.len() != num_grad.len() {
            return Err(format!(
                "grad len mismatch input {}: {} vs {}",
                idx,
                autograd_grad.len(),
                num_grad.len()
            ));
        }
        for (j, (&a, &n)) in autograd_grad.iter().zip(num_grad.iter()).enumerate() {
            let diff = (a - n).abs();
            if diff > atol && diff > rtol * n.abs().max(1e-8) {
                return Err(format!(
                    "input {} elem {}: autograd {} vs numerical {}",
                    idx, j, a, n
                ));
            }
        }
    }
    Ok(())
}
