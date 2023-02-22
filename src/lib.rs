extern crate core;

mod weights;
mod moran;
mod geary;
mod join_counts;
mod binarized;
mod neighbors;
mod utils;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn sway(py: Python, m: &PyModule) -> PyResult<()> {
    weights::register(py, m)?;
    join_counts::register(py, m)?;
    Ok(())
}
