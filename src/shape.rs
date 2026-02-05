//! Shape management for tensors: dimensions and layout.

use std::fmt;
use thiserror::Error;

/// Error when shape is invalid for an operation.
#[derive(Error, Debug)]
#[error("shape error: {0}")]
pub struct ShapeError(pub String);

/// Shape of a tensor: ordered list of dimension sizes.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a shape from dimension sizes.
    pub fn new(dims: Vec<usize>) -> Self {
        Shape { dims }
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Dimension sizes as slice.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Check if this shape is compatible for element-wise op with another (same shape).
    pub fn same_as(&self, other: &Shape) -> bool {
        self.dims == other.dims
    }

    /// Check if shape is a scalar (0 dimensions or single element).
    pub fn is_scalar(&self) -> bool {
        self.numel() <= 1
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape{:?}", self.dims)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_numel() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.rank(), 3);
    }
}
