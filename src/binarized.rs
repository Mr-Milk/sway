use itertools::Itertools;
use ndarray::{Array1, ArrayView1};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn threshold_ostu(x: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let x = x.as_array();
    Ok(1.0)
}

fn ostu_level(arr: ArrayView1<f64>) -> f64 {
    1.0
}

// fn hist(arr: ArrayView1<f64>, nbins: usize) {
//     arr.iter()
//         .sorted_by(|a, b|
//             **a.partial_cmp(**b).unwrap()
//         )
// }


#[pyfunction]
fn threshold_mean(x: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
    let x = x.as_array();
    let all_thresh = x.outer_iter()
        .into_par_iter()
        .map(|row| row.sum() / (row.len() as f64))
        .collect();
    Ok(all_thresh)
}

