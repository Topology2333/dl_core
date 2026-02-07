//! Test state_dict save/load: train or init a model, save, load into new model, assert same forward.

use dl_core::{load_state_dict, save_state_dict, set_seed, CpuBackend, Linear, Module, Shape, Tensor};
use std::sync::Arc;

#[test]
fn test_state_dict_save_load_linear() {
    set_seed(99);
    let backend = Arc::new(CpuBackend::new());

    let mut model = Linear::new(2, 1, backend.clone()).unwrap();
    model.init_xavier().unwrap();

    let x = Tensor::from_vec(vec![1.0f32, 2.0], Shape::new(vec![1, 2]), backend.clone()).unwrap();
    let out_before = model.forward(&x).unwrap();
    let out_before_data = out_before.data().to_vec();

    let states = model.state_dict();
    let path = std::env::temp_dir().join("dl_core_state_dict_test.json");
    save_state_dict(&path, &states).unwrap();

    let loaded_states = load_state_dict(&path).unwrap();
    let mut model2 = Linear::new(2, 1, backend.clone()).unwrap();
    model2.load_state_dict(&loaded_states, backend.clone()).unwrap();

    let out_after = model2.forward(&x).unwrap();
    let out_after_data = out_after.data().to_vec();

    assert_eq!(out_before_data.len(), out_after_data.len());
    for (a, b) in out_before_data.iter().zip(out_after_data.iter()) {
        assert!((a - b).abs() < 1e-5, "forward mismatch: {} vs {}", a, b);
    }

    let _: Result<(), _> = std::fs::remove_file(&path);
}
