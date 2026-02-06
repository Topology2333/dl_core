//! End-to-end training: linear regression y = 2*x1 + 3*x2 + bias + noise.
//! Verifies full pipeline: data -> model -> loss -> backward -> optimizer.

use dl_core::data::{DataLoader, InMemoryDataset};
use dl_core::optimizer::SGD;
use dl_core::train::Trainer;
use dl_core::{set_seed, with_rng, CpuBackend, Linear, Shape};
use rand::Rng;
use std::sync::Arc;

const TRUE_W: [f32; 2] = [2.0, 3.0];
const TRUE_B: f32 = 1.0;
const N_SAMPLES: usize = 100;
const EPOCHS: usize = 150;
const LR: f32 = 0.02;

fn make_dataset(backend: &Arc<CpuBackend>) -> Result<InMemoryDataset, dl_core::TensorError> {
    set_seed(42);
    let mut samples = Vec::with_capacity(N_SAMPLES);
    for _ in 0..N_SAMPLES {
        let (x1, x2, noise) = with_rng(|rng| {
            let x1 = rng.gen_range(-1.0f32..=1.0);
            let x2 = rng.gen_range(-1.0f32..=1.0);
            let noise = rng.gen_range(-0.1f32..=0.1);
            (x1, x2, noise)
        });
        let y = TRUE_W[0] * x1 + TRUE_W[1] * x2 + TRUE_B + noise;

        let input = dl_core::Tensor::from_vec(
            vec![x1, x2],
            Shape::new(vec![1, 2]),
            backend.clone(),
        )?;
        let target = dl_core::Tensor::from_vec(
            vec![y],
            Shape::new(vec![1, 1]),
            backend.clone(),
        )?;
        samples.push((input, target));
    }
    Ok(InMemoryDataset::new(samples))
}

#[test]
fn test_linear_regression_loss_decreases() {
    set_seed(123);
    let backend = Arc::new(CpuBackend::new());
    let dataset = make_dataset(&backend).unwrap();
    let mut dataloader = DataLoader::new(dataset, 8);

    let mut model = Linear::new(2, 1, backend.clone()).unwrap();
    model.init_xavier().unwrap();
    let opt = SGD::new(LR);
    let mut trainer = Trainer::new(model, opt);

    let mut initial_loss = None::<f32>;
    let mut final_loss = None::<f32>;

    for _ in 0..EPOCHS {
        dataloader.reset();
        let (avg_loss, _) = trainer.run_epoch(backend.clone(), &mut dataloader).unwrap();
        if initial_loss.is_none() {
            initial_loss = Some(avg_loss);
        }
        final_loss = Some(avg_loss);
    }

    let init = initial_loss.unwrap();
    let fin = final_loss.unwrap();
    assert!(fin < init, "loss should decrease: initial {} final {}", init, fin);
}

#[test]
fn test_linear_regression_learns_weights() {
    set_seed(456);
    let backend = Arc::new(CpuBackend::new());
    let dataset = make_dataset(&backend).unwrap();
    let mut dataloader = DataLoader::new(dataset, 16);

    let mut model = Linear::new(2, 1, backend.clone()).unwrap();
    model.init_xavier().unwrap();
    let opt = SGD::new(LR);
    let mut trainer = Trainer::new(model, opt);

    for _ in 0..EPOCHS {
        dataloader.reset();
        trainer.run_epoch(backend.clone(), &mut dataloader).unwrap();
    }

    let w = trainer.model.weight.data();
    let b = trainer.model.bias.data();
    let wd = w.data();
    let bd = b.data();
    assert_eq!(wd.len(), 2);
    assert_eq!(bd.len(), 1);
    assert!((wd[0] - TRUE_W[0]).abs() < 0.5, "weight[0] {} ~ {}", wd[0], TRUE_W[0]);
    assert!((wd[1] - TRUE_W[1]).abs() < 0.5, "weight[1] {} ~ {}", wd[1], TRUE_W[1]);
    assert!((bd[0] - TRUE_B).abs() < 0.5, "bias {} ~ {}", bd[0], TRUE_B);
}
