//! Save/load state_dict (Vec<ParameterState>) to/from JSON files.

use crate::parameter::ParameterState;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Save state dict to a JSON file.
pub fn save_state_dict(path: impl AsRef<Path>, states: &[ParameterState]) -> Result<(), std::io::Error> {
    let f = File::create(path)?;
    let w = BufWriter::new(f);
    serde_json::to_writer(w, states).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Load state dict from a JSON file.
pub fn load_state_dict(path: impl AsRef<Path>) -> Result<Vec<ParameterState>, std::io::Error> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    serde_json::from_reader(r).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
