# dl_core

A layered, extensible deep learning core in Rust: autograd, parameters, and training loop. The design follows clear layer boundaries so that replacing or extending one layer (e.g. CPU → SIMD/GPU) does not affect the rest of the system.

## Layers

- **Storage (numerical)**: `Tensor`, `Shape`, `Backend`. Tensor holds data and shape; all ops (matmul, add, relu) go through the `Backend` trait so implementations can be swapped.
- **Autograd**: Computation graph, nodes, backward pass. Operators are first-class (Op trait + registry); adding a new op = implement + register, no engine changes.
- **NN**: `Module`, `Layer`, `Linear`, `ReLU`, `Sigmoid`, loss (`mse`, `mse_graph`). Parameters are distinct from intermediate tensors.
- **Training**: `Trainer`, `Optimizer` (e.g. SGD), `DataLoader`. Full loop: zero_grad → forward → loss → backward → optimizer step.

## Determinism

With the same input, same random seed, and same parameters, the implementation aims for the same output. Call `dl_core::set_seed(seed)` before model init or training. Initialization (e.g. Xavier) uses the thread-local RNG. In single-threaded CPU execution, reduce order is fixed for reproducibility. Exceptions may apply when using future backends (e.g. GPU) or third-party code.

## Usage

Minimal training loop: build dataset, model, optimizer, then run epochs.

```rust
use dl_core::{CpuBackend, Linear, SGD, Trainer, set_seed};
use dl_core::data::{InMemoryDataset, DataLoader};
use std::sync::Arc;

set_seed(42);
let backend = Arc::new(CpuBackend::new());
let mut model = Linear::new(2, 1, backend.clone()).unwrap();
model.init_xavier().unwrap();
let mut opt = SGD::new(0.02);
let mut trainer = Trainer::new(model, opt);
// ... build InMemoryDataset, then:
// dataloader.reset();
// let (avg_loss, n) = trainer.run_epoch(backend.clone(), &mut dataloader)?;
```

A full runnable example is the **linear regression** end-to-end test: fit `y = 2*x1 + 3*x2 + bias` with MSE. Run it with:

```bash
cargo test test_linear_regression
```

See [tests/train_linear_regression.rs](tests/train_linear_regression.rs) for the complete code (dataset construction, multiple epochs, and assertions on loss decrease and learned weights).

## Tests and gradient check

- **Gradient checks**: `tests/grad_check.rs` compares autograd with finite differences (`autograd::check::check_gradients`).
- **End-to-end training**: `tests/train_linear_regression.rs` runs a full training loop and asserts loss decrease and that the model learns the true weights.

## License

MIT.
