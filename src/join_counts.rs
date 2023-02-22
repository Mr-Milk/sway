use ndarray::ArrayView1;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rand::thread_rng;
use rayon::prelude::*;
use crate::utils::{chi2_p, Shuffle};

use crate::weights::SpatialWeights;

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_join_counts, m)?)?;
    Ok(())
}

// ww, wb, bw, bb, p_chi2, autocorr_pos, autocorr_neg,
// option: p_autocorr_pos, p_autocorr_neg, p_sim_chi2
type JoinCountPyResult = (
    f64, f64, f64, f64, f64, f64, f64,
    Option<f64>, Option<f64>, Option<f64>,
);

#[pyfunction]
#[pyo3(name = "_py_join_counts")]
fn py_join_counts(
    ys: PyReadonlyArray2<bool>,
    w: &mut SpatialWeights,
    permutations: usize,
    alpha: f64,
    early_stop: bool,
) -> Vec<JoinCountPyResult> {
    let ys = ys.as_array();
    w.set_transform("B").expect("Enforce binary transformation failed.");
    let adj_list = w.to_adjlist_index(Some(true));
    ys.outer_iter()
        .into_par_iter()
        .map(|row| {
            let ac = area_count(row, &adj_list);

            let mut run_permutation = true;
            if permutations == 0 {
                run_permutation = false;
            }
            // don't run permutation if chi2-square test is non-significance
            if early_stop & (ac.pvalue > alpha) {
                run_permutation = false;
            }

            if run_permutation {
                let mut shuffle_exp = row.to_owned();
                let mut rng = thread_rng();
                let mut sim_chi2 = Vec::with_capacity(permutations);
                let mut sim_autocorr_pos = Vec::with_capacity(permutations);
                let mut sim_autocorr_neg = Vec::with_capacity(permutations);

                (0..permutations).for_each(|_| {
                    shuffle_exp.shuffle(&mut rng);
                    let sim_ac = area_count(
                        shuffle_exp.view(),
                        &adj_list);
                    sim_chi2.push(sim_ac.chi2);
                    sim_autocorr_pos.push(sim_ac.bb);
                    sim_autocorr_neg.push(sim_ac.bw + sim_ac.wb);
                });

                let p_sim_chi2 = pseudo_p(&sim_chi2, ac.chi2, permutations);
                let p_sim_autocorr_pos = pseudo_p(&sim_autocorr_pos, ac.autocorr_pos, permutations);
                let p_sim_autocorr_neg = pseudo_p(&sim_autocorr_neg, ac.autocorr_neg, permutations);
                (ac.ww, ac.wb, ac.bw, ac.bb, ac.pvalue, ac.autocorr_pos, ac.autocorr_neg,
                 Some(p_sim_autocorr_pos), Some(p_sim_autocorr_neg), Some(p_sim_chi2))
            } else {
                (ac.ww, ac.wb, ac.bw, ac.bb, ac.pvalue, ac.autocorr_pos, ac.autocorr_neg,
                 None, None, None)
            }

        })
        .collect()
}

struct AreaCountsResult {
    bb: f64,
    ww: f64,
    bw: f64,
    wb: f64,
    chi2: f64,
    pvalue: f64,
    autocorr_pos: f64,
    autocorr_neg: f64,
}

fn area_count(
    y: ArrayView1<bool>,
    adj_list: &Vec<(usize, usize)>,
) -> AreaCountsResult {
    let (mut bb, mut ww, mut bw, mut wb) = (0, 0, 0, 0);
    for (focal, neighbor) in adj_list {
        // UNSAFE get
        let a = y[*focal];
        let b = y[*neighbor];
        match (a, b) {
            (true, true) => bb += 1,
            (false, false) => ww += 1,
            (true, false) => bw += 1,
            (false, true) => wb += 1,
        }
    }

    let ww = (ww as f64) / 2.0;
    let wb = (wb as f64) / 2.0;
    let bb = (bb as f64) / 2.0;
    let bw = (bw as f64) / 2.0;
    let autocorr_neg = bb;
    let autocorr_pos = bw + wb;
    let (chi2, pvalue) = chi2_22table(ww, wb, bw, bb);

    AreaCountsResult {
        ww,
        wb,
        bb,
        bw,
        autocorr_pos,
        autocorr_neg,
        chi2,
        pvalue
    }
}

fn chi2_22table(a1: f64, a2: f64, b1: f64, b2: f64) -> (f64, f64) {
    let y1 = a1 + a2;
    let y2 = b1 + b2;
    let x1 = a1 + b1;
    let x2 = a2 + b2;
    let sum = x1 + x2;
    let e11 = x1 * y1 / sum;
    let e12 = y1 * x2 / sum;
    let e21 = y2 * x1 / sum;
    let e22 = y2 * x2 / sum;
    let chi2 = [a1, a2, b1, b2].into_iter()
        .zip([e11, e12, e21, e22].into_iter())
        .fold(0.0, |acc, (o, e)| {
            // apply yates's correction since ddof is 1 for 2*2 table
            let v = ((o - e).abs() - 0.5).powi(2) / e;
            acc + v
        });
    let pvalue = chi2_p(chi2, 1.0);
    // let chi2_dist = ChiSquared::new(1.0).unwrap();
    // let pvalue = 1.0 - chi2_dist.cdf(chi2);
    (chi2, pvalue)
}

fn pseudo_p(sim: &Vec<f64>, real: f64, permutations: usize) -> f64 {
    let larger = sim.iter().fold(0.0, |acc, i| {
        if *i >= real {
            acc + 1.0
        } else {
            acc
        }
    });
    larger + 1.0 / (permutations as f64) + 1.0
}

#[cfg(test)]
mod tests {
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    #[test]
    fn chi2() {
        let chi = ChiSquared::new(1.0).unwrap();
        assert!((0.6826894921370859 - chi.cdf(1.0)).abs() < 0.0000001)
    }
}
