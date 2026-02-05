//! Layer trait: alias for Module for single-layer components.

use super::module::Module;

/// Layer: a Module (e.g. Linear, ReLU). All layers implement Module.
pub trait Layer: Module {}
