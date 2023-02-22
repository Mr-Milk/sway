#![allow(non_snake_case)]

use ndarray::{Array, ArrayView1};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rand::thread_rng;
use rayon::prelude::*;

use crate::utils::{zscore_p, Shuffle};
use crate::weights::SpatialWeights;

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_moran, m)?)?;
    Ok(())
}

// I, EI, p_norm,
type MoranPyResult = (f64, f64, f64, Option<f64>, Option<f64>);


#[pyfunction]
fn py_moran(
    ys: PyReadonlyArray2<f64>,
    w: &mut SpatialWeights,
    two_tailed: bool,
    permutations: usize,
    alpha: f64,
    early_stop: bool,
) -> Vec<MoranPyResult> {
    let ys = ys.as_array();
    w.set_transform("R").expect("Enforcing row transformation failed.");

    // call this three to ensure it's calculated.
    w.s0();
    w.s1();
    w.s2();

    ys.outer_iter()
        .into_par_iter()
        .map(|row| {
            let mm = moments(row, w);
            let s0 = w.s0.unwrap();

            let wx: f64 = w.wx_i(row);
            let I = mm.n / s0 * wx / mm.z2ss;

            let z_norm = (I - mm.EI) / mm.seI_norm;
            let z_rand = (I - mm.EI) / mm.seI_rand;

            let p_norm = zscore_p(z_norm, two_tailed);
            let p_rand = zscore_p(z_rand, two_tailed);

            let mut run_permutation = true;
            if permutations == 0 {
                run_permutation = false;
            }

            if early_stop & ((p_norm > alpha) | (p_rand > alpha)) {
                run_permutation = false;
            }

            if run_permutation {
                let mut shuffle_exp = row.to_owned();
                let mut rng = thread_rng();

                let mut larger: usize = 0;
                let sim: Vec<f64> = (0..permutations).map(|_| {
                    shuffle_exp.shuffle(&mut rng);
                    let wx: f64 = w.wx_i(shuffle_exp.view());
                    let sim_I = (mm.n / s0) * (wx / mm.z2ss);
                    if sim_I >= I {
                        larger += 1
                    };
                    sim_I
                }).collect();
                let sim = Array::from_vec(sim);

                if (permutations - larger) < larger {
                    larger = permutations - larger;
                }

                let p_sim = (larger as f64 + 1.0) / (permutations as f64 + 1.0);
                let EI_sim = sim.sum() / permutations as f64;
                let seI_sim = sim.std(0.0);
                let z_sim = (I - EI_sim) / seI_sim;
                let p_z_sim = zscore_p(z_sim, false);
                return (I, p_norm, p_rand, Some(p_sim), Some(p_z_sim));
            }
            return (I, p_norm, p_rand, None, None);
        }).collect()
}


    struct MoranMoments {
        n: f64,
        z2ss: f64,
        EI: f64,
        VI_norm: f64,
        seI_norm: f64,
        VI_rand: f64,
        seI_rand: f64,
    }

    fn moments(
        x: ArrayView1<f64>,
        w: &SpatialWeights,
    ) -> MoranMoments {
        let n = x.len() as f64;
        let s0 = w.s0.unwrap();
        let s1 = w.s1.unwrap();
        let s2 = w.s2.unwrap();
        let s02 = s0 * s0;

        let mean_x = x.mean().unwrap();
        let z = x.to_owned() - mean_x;
        let z2ss = (&z * &z).sum();

        let EI = -1.0 / (n - 1.0);

        let n2 = n * n;
        let v_num = n2 * s1 - n * s2 + 3.0 * s02;
        let v_den = (n - 1.0) * (n + 1.0) * s02;
        let VI_norm = v_num / v_den - (1.0 / (n - 1.0)).powi(2);
        let seI_norm = VI_norm.powf(1.0 / 2.0);

        // variance under randomization
        let xd4 = z.mapv(|a| a.powi(4));
        let xd2 = z.mapv(|a| a.powi(2));
        let k_num = xd4.sum() / n;
        let k_den = (xd2.sum() / n).powi(2);
        let k = k_num / k_den;

        let A = n * ((n2 - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s02);
        let B = k * ((n2 - n) * s1 - 2.0 * n * s2 + 6.0 * s02);
        let VIR = (A - B) / ((n - 1.0) * (n - 2.0) * (n - 3.0) * s02) - EI * EI;

        MoranMoments {
            n,
            z2ss,
            EI,
            VI_norm,
            seI_norm,
            VI_rand: VIR,
            seI_rand: VIR.powf(0.5),
        }
    }