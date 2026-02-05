//! Numerical gradient check tests: compare autograd with finite difference.

use dl_core::autograd::check::{check_gradients, numerical_grad, DEFAULT_EPS};
use dl_core::autograd::{Graph, NodeId};
use dl_core::CpuBackend;
use dl_core::tensor::Tensor;
use dl_core::Shape;
use std::sync::Arc;

fn backend() -> Arc<CpuBackend> {
    Arc::new(CpuBackend::new())
}

#[test]
fn test_numerical_grad_add() {
    let b = backend();
    let x = Tensor::from_vec(vec![1.0, 2.0], Shape::new(vec![2]), b.clone()).unwrap();
    let f = |t: &Tensor| t.data().iter().sum::<f32>();
    let g = numerical_grad(&x, f, DEFAULT_EPS);
    assert_eq!(g.len(), 2);
    assert!((g[0] - 1.0).abs() < 1e-2, "g[0] = {} expected ~1", g[0]);
    assert!((g[1] - 1.0).abs() < 1e-2, "g[1] = {} expected ~1", g[1]);
}

#[test]
fn test_check_gradients_add() {
    let b = backend();
    let x = Tensor::from_vec(vec![1.0, 2.0], Shape::new(vec![2]), b.clone()).unwrap();
    let y = Tensor::from_vec(vec![3.0, 4.0], Shape::new(vec![2]), b.clone()).unwrap();
    let build = |g: &mut Graph, ids: &[NodeId]| {
        assert_eq!(ids.len(), 2);
        g.add(ids[0], ids[1]).and_then(|sum_id| g.sum(sum_id))
    };
    check_gradients(&build, &[x, y], DEFAULT_EPS, 1e-2, 1e-2).unwrap();
}

#[test]
fn test_check_gradients_matmul() {
    let b = backend();
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]), b.clone()).unwrap();
    let c = Tensor::from_vec(vec![0.5, 0.5, 0.5, 0.5], Shape::new(vec![2, 2]), b.clone()).unwrap();
    let build = |g: &mut Graph, ids: &[NodeId]| {
        assert_eq!(ids.len(), 2);
        g.matmul(ids[0], ids[1]).and_then(|out_id| g.sum(out_id))
    };
    check_gradients(&build, &[a, c], DEFAULT_EPS, 1e-2, 1e-2).unwrap();
}

#[test]
fn test_check_gradients_relu() {
    let b = backend();
    // Avoid exact 0: ReLU'(0) is undefined (subgradient); we use 0, numerical central diff gives 0.5.
    let x = Tensor::from_vec(vec![-1.0, 0.5, 0.01, 2.0], Shape::new(vec![4]), b.clone()).unwrap();
    let build = |g: &mut Graph, ids: &[NodeId]| {
        assert_eq!(ids.len(), 1);
        g.relu(ids[0]).and_then(|out_id| g.sum(out_id))
    };
    check_gradients(&build, &[x], DEFAULT_EPS, 1e-2, 1e-2).unwrap();
}
