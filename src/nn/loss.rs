//! Loss functions: MSE, etc. Input (pred, target), output scalar Tensor.

use crate::autograd::{Graph, NodeId};
use crate::tensor::Tensor;

/// MSE: mean squared error. (pred - target)^2, then mean. Scalar output.
pub fn mse(pred: &Tensor, target: &Tensor) -> crate::TensorResult<Tensor> {
    if !pred.shape().same_as(target.shape()) {
        return Err(crate::TensorError::Shape(crate::ShapeError(
            "mse: pred and target shape mismatch".into(),
        )));
    }
    let neg_target = target.scale(-1.0)?;
    let diff = pred.add(&neg_target)?;
    let sq = diff.mul(&diff)?;
    let sum = sq.sum()?;
    let n = pred.numel() as f32;
    sum.scale(1.0 / n)
}

/// MSE in graph: loss_id = mse_graph(g, pred_id, target_tensor).
/// Builds (pred - target)^2, sum, scale by 1/n. Returns loss node id.
pub fn mse_graph(
    g: &mut Graph,
    pred_id: NodeId,
    target: &Tensor,
) -> crate::GraphResult<NodeId> {
    let target_id = g.var(target.clone());
    let diff_id = g.sub(pred_id, target_id)?;
    let n = g.data(pred_id)?.numel() as f32;
    let backend = g.data(pred_id)?.backend();
    let sq_id = g.mul(diff_id, diff_id)?;
    let sum_id = g.sum(sq_id)?;
    let constant = Tensor::from_vec(
        vec![1.0 / n],
        crate::shape::Shape::new(vec![1]),
        backend,
    )
    .map_err(|e| crate::GraphError(e.to_string()))?;
    let c_id = g.var(constant);
    let scale_id = g.mul(sum_id, c_id)?;
    Ok(scale_id)
}

/// Cross-entropy in graph: logits_id (logits [B, C]), target one-hot [B, C].
/// Loss = -mean over batch of sum_over_C(target * log(softmax(logits))).
pub fn ce_graph(
    g: &mut Graph,
    logits_id: NodeId,
    target: &Tensor,
) -> crate::GraphResult<NodeId> {
    let (b, backend) = {
        let logits_data = g.data(logits_id)?;
        if !logits_data.shape().same_as(target.shape()) {
            return Err(crate::GraphError(
                "ce_graph: logits and target shape mismatch".into(),
            ));
        }
        let b = logits_data.shape().dims()[0] as f32;
        let backend = logits_data.backend();
        (b, backend)
    };
    let softmax_id = g.softmax(logits_id)?;
    let log_id = g.log(softmax_id)?;
    let target_id = g.var(target.clone());
    let mul_id = g.mul(target_id, log_id)?;
    let sum_id = g.sum(mul_id)?;
    let constant = Tensor::from_vec(
        vec![-1.0 / b],
        crate::shape::Shape::new(vec![1]),
        backend,
    )
    .map_err(|e| crate::GraphError(e.to_string()))?;
    let c_id = g.var(constant);
    let loss_id = g.mul(sum_id, c_id)?;
    Ok(loss_id)
}
