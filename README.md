# dl_core

A layered, extensible deep learning core in Rust: autograd, parameters, and training loop. The design follows clear layer boundaries so that replacing or extending one layer (e.g. CPU → SIMD/GPU) does not affect the rest of the system.

## Layers

- **Storage (numerical)**: `Tensor`, `Shape`, `Backend`. Tensor holds data and shape; all ops (matmul, add, relu) go through the `Backend` trait so implementations can be swapped.
- **Autograd**: Computation graph, nodes, backward pass. Operators are first-class (Op trait + registry); adding a new op = implement + register, no engine changes.
- **NN**: `Module`, `Layer`, `Linear`, `ReLU`, `Sigmoid`, loss (`mse`, `mse_graph`). Parameters are distinct from intermediate tensors.
- **Training**: `Trainer`, `Optimizer` (e.g. SGD), `DataLoader`. Full loop: zero_grad → forward → loss → backward → optimizer step.

## Determinism

With the same input, same random seed, and same parameters, the implementation aims for the same output. Call `dl_core::set_seed(seed)` before model init or training. Initialization (e.g. Xavier) uses the thread-local RNG. In single-threaded CPU execution, reduce order is fixed for reproducibility. Exceptions may apply when using future backends (e.g. GPU) or third-party code.

## Usage sketch

```rust
use dl_core::{CpuBackend, Linear, SGD, Trainer, set_seed, mse_graph};
use std::sync::Arc;

set_seed(42);
let backend = Arc::new(dl_core::CpuBackend::new());
let mut model = Linear::new(2, 1, backend.clone()).unwrap();
model.init_xavier().unwrap();
let mut opt = SGD::new(0.01);
let mut trainer = Trainer::new(model, opt);
// trainer.step(backend, &input, &target);
```

## Tests and gradient check

Numerical gradient checks compare autograd gradients with finite differences. See `tests/grad_check.rs` and `autograd::check::check_gradients`.

## License

MIT.
