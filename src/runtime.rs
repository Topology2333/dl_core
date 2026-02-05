//! Runtime: global seed for deterministic behavior.
//! Same input, same seed, same parameters -> same output.

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cell::RefCell;

thread_local! {
    static RNG: RefCell<Option<StdRng>> = RefCell::new(None);
}

/// Set the global random seed for this thread. Call before model init or training
/// to get reproducible results. Same seed + same code path -> same outputs.
pub fn set_seed(seed: u64) {
    RNG.with(|rng| {
        *rng.borrow_mut() = Some(StdRng::seed_from_u64(seed));
    });
}

/// Run closure with the thread-local RNG (initialized from seed 0 if not set).
pub fn with_rng<F, T>(f: F) -> T
where
    F: FnOnce(&mut StdRng) -> T,
{
    RNG.with(|rng| {
        let mut opt = rng.borrow_mut();
        if opt.is_none() {
            *opt = Some(StdRng::seed_from_u64(0));
        }
        f(opt.as_mut().unwrap())
    })
}
