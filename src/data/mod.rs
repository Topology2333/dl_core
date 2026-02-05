//! Data pipeline: Dataset trait and DataLoader for batching.

use crate::tensor::Tensor;

/// Dataset: indexed collection of (input, target) pairs.
pub trait Dataset {
    /// Number of samples.
    fn len(&self) -> usize;

    /// Get sample at index. Returns (input_tensor, target_tensor).
    fn get(&self, index: usize) -> Option<(Tensor, Tensor)>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Simple DataLoader: iterates over batches. No shuffling in v1.
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    index: usize,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        DataLoader {
            dataset,
            batch_size,
            index: 0,
        }
    }

    /// Next batch: (inputs, targets). inputs shape [batch, ...], targets shape [batch, ...].
    /// Returns None when no more full batch.
    pub fn next_batch(&mut self) -> Option<(Vec<Tensor>, Vec<Tensor>)> {
        let start = self.index;
        if start >= self.dataset.len() {
            return None;
        }
        let end = (start + self.batch_size).min(self.dataset.len());
        let mut inputs = Vec::with_capacity(end - start);
        let mut targets = Vec::with_capacity(end - start);
        for i in start..end {
            if let Some((x, y)) = self.dataset.get(i) {
                inputs.push(x);
                targets.push(y);
            }
        }
        self.index = end;
        if inputs.is_empty() {
            None
        } else {
            Some((inputs, targets))
        }
    }

    /// Reset to start of dataset.
    pub fn reset(&mut self) {
        self.index = 0;
    }
}

/// In-memory dataset from vec of (input, target).
pub struct InMemoryDataset {
    samples: Vec<(Tensor, Tensor)>,
}

impl InMemoryDataset {
    pub fn new(samples: Vec<(Tensor, Tensor)>) -> Self {
        InMemoryDataset { samples }
    }
}

impl Dataset for InMemoryDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<(Tensor, Tensor)> {
        self.samples.get(index).cloned()
    }
}
