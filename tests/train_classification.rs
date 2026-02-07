//! End-to-end classification: MLP2 + CE loss. Synthetic 2D two-class data.

use dl_core::data::{DataLoader, InMemoryDataset};
use dl_core::optimizer::SGD;
use dl_core::train::Trainer;
use dl_core::{
    set_seed, with_rng, CpuBackend, MLP2, Shape, Tensor,
};
use rand::Rng;
use std::sync::Arc;

const N_SAMPLES: usize = 200;
const EPOCHS: usize = 80;
const LR: f32 = 0.05;
const BATCH_SIZE: usize = 16;

/// Generate 2D points and binary labels: class 0 near (-1,-1), class 1 near (1,1).
fn make_dataset(backend: &Arc<CpuBackend>) -> Result<InMemoryDataset, dl_core::TensorError> {
    set_seed(42);
    let mut samples = Vec::with_capacity(N_SAMPLES);
    for _ in 0..N_SAMPLES {
        let (x1, x2, label) = with_rng(|rng| {
            let c = rng.gen_range(0..2);
            let x1 = if c == 0 {
                rng.gen_range(-1.5f32..-0.3)
            } else {
                rng.gen_range(0.3f32..1.5)
            };
            let x2 = if c == 0 {
                rng.gen_range(-1.5f32..-0.3)
            } else {
                rng.gen_range(0.3f32..1.5)
            };
            (x1, x2, c)
        });
        // Shape [2] so that stack yields [B, 2] for matmul.
        let input =
            Tensor::from_vec(vec![x1, x2], Shape::new(vec![2]), backend.clone())?;
        let target = Tensor::from_vec(
            if label == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] },
            Shape::new(vec![2]),
            backend.clone(),
        )?;
        samples.push((input, target));
    }
    Ok(InMemoryDataset::new(samples))
}

#[test]
fn test_mlp_classification_ce() {
    set_seed(123);
    let backend = Arc::new(CpuBackend::new());
    let dataset = make_dataset(&backend).unwrap();
    let mut dataloader = DataLoader::new(dataset, BATCH_SIZE);

    let mut model = MLP2::new(2, 8, 2, backend.clone()).unwrap();
    model.init_xavier().unwrap();
    let opt = SGD::new(LR);
    let mut trainer = Trainer::new(model, opt);

    let mut initial_loss = None::<f32>;
    let mut final_loss = None::<f32>;

    for _epoch in 0..EPOCHS {
        dataloader.reset();
        while let Some((inputs, targets)) = dataloader.next_batch() {
            let input_batch = Tensor::stack(&inputs, 0).unwrap();
            let target_batch = Tensor::stack(&targets, 0).unwrap();
            let r = trainer
                .step_batch_ce(backend.clone(), &input_batch, &target_batch)
                .unwrap();
            if initial_loss.is_none() {
                initial_loss = Some(r.loss);
            }
            final_loss = Some(r.loss);
        }
    }

    let init = initial_loss.unwrap();
    let fin = final_loss.unwrap();
    eprintln!(
        "  classification CE loss: initial={:.4} -> final={:.4} (epochs={})",
        init, fin, EPOCHS
    );
    assert!(fin < init, "CE loss should decrease: {} -> {}", init, fin);
}
