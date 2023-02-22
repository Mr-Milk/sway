use std::collections::{HashMap, HashSet};

use nalgebra_sparse::CsrMatrix;
use nalgebra_sparse::ops::Op;
use nalgebra_sparse::ops::serial::{spadd_csr_prealloc, spadd_pattern};
use ndarray::prelude::*;
use pyo3::{pyclass, pymethods, PyResult, Python};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::utils::default_arg;

type Neighbors = Vec<(usize, Vec<usize>)>;

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SpatialWeights>()?;
    Ok(())
}

struct PointNeighbors {
    nn: usize,
    focal: usize,
    neighbors: Vec<usize>,
    weights: Vec<f64>,
}


#[derive(PartialEq)]
enum Transform {
    R,
    // Row transform
    B, // Binary transform
}

// subclass allow this to be inherited
#[pyclass(subclass)]
pub struct SpatialWeights {
    #[allow(dead_code)]
    n: usize,
    transform: Transform,
    pub id_order: Vec<usize>,
    pub id2i: HashMap<usize, usize>,
    neighbors_list: Vec<PointNeighbors>,
    row_index: Vec<usize>,
    col_index: Vec<usize>,
    w_sparse: CsrMatrix<f64>,
    pub(crate) s0: Option<f64>,
    pub(crate) s1: Option<f64>,
    pub(crate) s2: Option<f64>,
}

#[pymethods]
impl SpatialWeights {
    #[new]
    fn from_neighbors(neighbors: Vec<Vec<usize>>, labels: Vec<usize>) -> Self {
        let n = neighbors.len();
        let id2i: HashMap<usize, usize> =
            labels.iter()
                .enumerate()
                .map(|(order, id)| {
                    (*id, order)
                }).collect();
        // let mut id_order: Vec<usize> = labels;

        let mut ptr: usize = 0;
        let mut indptr = vec![0]; // row_offsets
        let mut indice = vec![]; // col_index
        let mut row_index = vec![];
        let mut w_data = vec![];
        let mut neighbors_list: Vec<PointNeighbors> = vec![];

        for (ix, (focal, neighs)) in labels.iter().zip(neighbors).enumerate() {
            let nn = neighs.len();
            // create weight matrix
            let weights = vec![1.0; nn];
            // println!("weights: {:?}", weights.clone());
            w_data.extend(weights.iter());

            ptr += nn;
            indptr.push(ptr);

            let mut unsorted_indice = vec![];
            for n in neighs.iter() {
                let ordered_id = *id2i.get(n).unwrap();
                row_index.push(ix);
                unsorted_indice.push(ordered_id);
            }
            unsorted_indice.sort();
            indice.extend(unsorted_indice.iter());

            let pn = PointNeighbors {
                nn,
                focal: *focal,
                neighbors: neighs.to_owned(),
                weights,
            };
            neighbors_list.push(pn);
        };

        // println!("{:?}\n {:?}\n {:?}\n {:?}\n {:?}", n, indptr.clone(), row_index.clone(), indice.clone(), w_data.clone());

        let w_sparse = CsrMatrix::try_from_csr_data(n, n,
                                                    indptr,
                                                    indice.to_owned(),
                                                    w_data).unwrap();

        SpatialWeights {
            n,
            id_order: labels,
            id2i,
            neighbors_list,
            transform: Transform::B,
            row_index,
            col_index: indice,
            w_sparse,
            s0: None,
            s1: None,
            s2: None,
        }
    }

    fn _reset(&mut self) {
        self.s0 = None;
        self.s1 = None;
        self.s2 = None;
        let mut w_data: Vec<f64> = vec![];
        for neighbors in &mut self.neighbors_list {
            let nn = neighbors.nn;
            let new_weights = match self.transform {
                Transform::B => vec![1.0; nn],
                Transform::R => vec![1.0 / (nn as f64); nn],
            };
            w_data.extend(new_weights.iter());
            neighbors.weights = new_weights;
        }

        let sparse_data = self.w_sparse.values_mut();
        for (ele, new_ele) in sparse_data.iter_mut().zip(w_data) {
            *ele = new_ele
        }
    }

    #[getter]
    fn get_transform(&self) -> PyResult<String> {
        match self.transform {
            Transform::B => Ok(String::from("B")),
            Transform::R => Ok(String::from("R")),
        }
    }

    #[setter]
    pub(crate) fn set_transform(&mut self, value: &str) -> PyResult<()> {
        match value {
            "B" => {
                if self.transform != Transform::B {
                    self.transform = Transform::B;
                    self._reset();
                }
                Ok(())
            }
            "R" => {
                if self.transform != Transform::R {
                    self.transform = Transform::R;
                    self._reset();
                }
                Ok(())
            }
            _ => Err(PyValueError::new_err("Unsupported transformation, \
            support 'R' and 'B' currently."))
        }
    }

    #[getter]
    fn neighbors(&self) -> Neighbors {
        self.neighbors_list.iter().map(|neighbors| {
            (neighbors.focal, neighbors.neighbors.to_owned())
        }).collect()
    }

    #[getter]
    fn neighbors_index(&self) -> Neighbors {
        self.neighbors_list.iter().map(|neighbors| {
            let focal = *(self.id2i.get(&neighbors.focal).unwrap());
            let neighs: Vec<usize> = neighbors.neighbors.iter().map(|n| {
                *(self.id2i.get(n).unwrap())
            }).collect();
            (focal, neighs)
        }).collect()
    }

    fn _to_sparse(&self) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
        let (row_offsets, col_indices, values) = (&self.w_sparse).csr_data();
        return (values.to_vec(), col_indices.to_vec(), row_offsets.to_vec());
    }

    pub(crate) fn to_adjlist(&self, symmetric: Option<bool>) -> Vec<(usize, usize)> {
        let symmetric = default_arg(symmetric, true);

        return if symmetric {
            let mut adjlist = vec![];

            for neighbors in self.neighbors_list.iter() {
                for i in neighbors.neighbors.iter() {
                    adjlist.push((neighbors.focal, *i))
                }
            };
            adjlist
        } else {
            let mut adjlist: HashSet<(usize, usize)> = HashSet::new();

            for neighbors in self.neighbors_list.iter() {
                for i in neighbors.neighbors.iter() {
                    let pair = (neighbors.focal, *i);
                    if !adjlist.contains(&pair) {
                        if !adjlist.contains(&(pair.1, pair.0)) {
                            adjlist.insert(pair);
                        }
                    }
                }
            };
            adjlist.into_iter().collect()
        }
    }

    pub(crate) fn to_adjlist_index(&self, symmetric: Option<bool>) -> Vec<(usize, usize)> {
        let symmetric = default_arg(symmetric, true);

        return if symmetric {
            let mut adjlist = vec![];
            for neighbors in self.neighbors_list.iter() {
                let focal = self.id2i.get(&neighbors.focal).unwrap();
                for i in neighbors.neighbors.iter() {
                    let neigh = self.id2i.get(i).unwrap();
                    adjlist.push((*focal, *neigh))
                }
            };
            adjlist
        } else {
            let mut adjlist: HashSet<(usize, usize)> = HashSet::new();

            for neighbors in self.neighbors_list.iter() {
                let focal = self.id2i.get(&neighbors.focal).unwrap();
                for i in neighbors.neighbors.iter() {
                    let neigh = self.id2i.get(i).unwrap();
                    let pair = (*focal, *neigh);
                    if !adjlist.contains(&pair) {
                        if !adjlist.contains(&(pair.1, pair.0)) {
                            adjlist.insert(pair);
                        }
                    }
                }
            };
            adjlist.into_iter().collect()
        }
    }

    //
    // #[getter]
    // fn _py_adjlist_origin(&self) -> Vec<(usize, usize)> {
    //     let mut adjlist = vec![];
    //     for neighbors in self.neighbors_list.iter() {
    //         for i in neighbors.origin_neighbors.iter() {
    //             adjlist.push((neighbors.origin_focal, *i))
    //         }
    //     };
    //     adjlist
    // }

    #[getter]
    fn id2i(&self) -> PyResult<HashMap<usize, usize>> {
        Ok(self.id2i.clone())
    }

    #[getter]
    fn id_order(&self) -> PyResult<Vec<usize>> {
        Ok(self.id_order.clone())
    }

    fn _s0(&self) -> f64 {
        self.w_sparse.values().iter().sum()
    }

    fn _s1(&self) -> f64 {
        let w1_pattern = spadd_pattern(self.w_sparse.pattern(), self.w_sparse.transpose().pattern());
        let w1_len = w1_pattern.nnz();
        let mut w1 = CsrMatrix::try_from_pattern_and_values(w1_pattern, vec![0.0; w1_len]).unwrap();
        spadd_csr_prealloc(1.0, &mut w1, 1.0, Op::NoOp(&self.w_sparse)).unwrap();
        spadd_csr_prealloc(1.0, &mut w1, 1.0, Op::Transpose(&self.w_sparse)).unwrap();
        let w1_data: Array1<f64> = w1.values().iter().map(|i| *i).collect();
        let s1 = (&w1_data * &w1_data).sum() / 2.0;
        return s1;
    }

    fn _s2(&self) -> f64 {
        let w_sum0: Array1<f64> = self.w_sparse
            .transpose()
            .row_iter()
            .map(|row| row.values().iter().fold(0.0, |acc, a| acc + *a))
            .collect();
        let w_sum1: Array1<f64> = self.w_sparse
            .row_iter()
            .map(|row| row.values().iter().fold(0.0, |acc, a| acc + *a))
            .collect();
        let s2 = (&w_sum0 + &w_sum1).mapv(|a| a.powi(2)).sum();
        return s2;
    }

    #[getter]
    pub fn s0(&mut self) -> f64 {
        let s0 = match self.s0 {
            Some(v) => v,
            None => {
                let v = self._s0();
                self.s0 = Some(v);
                v
            }
        };
        s0
    }

    #[getter]
    pub fn s1(&mut self) -> f64 {
        let s1 = match self.s1 {
            Some(v) => v,
            None => {
                let v = self._s1();
                self.s1 = Some(v);
                v
            }
        };
        s1
    }

    #[getter]
    pub fn s2(&mut self) -> f64 {
        let s2 = match self.s2 {
            Some(v) => v,
            None => {
                let v = self._s2();
                self.s2 = Some(v);
                v
            }
        };
        s2
    }
}

// Methods that don't expose to Python side
impl SpatialWeights {
    /// Compute wx for moran
    pub fn wx_i(&self, z: ArrayView1<f64>) -> f64
    {
        self
            .w_sparse
            .row_iter()
            .enumerate()
            .map(|(ix, row)| {
                let pz: Array1<f64> = row.col_indices().iter().map(|i| z[*i]).collect();
                let r = Array::from(row.values().to_vec());
                (&r * &pz).sum() * z[ix]
            }).collect::<Array1<f64>>().sum()
    }

    /// Compute wx for geary
    pub fn wx_c(&self, z: ArrayView1<f64>) -> f64 {
        let w: Array1<f64> = Array::from_vec(self.w_sparse.values().to_vec());
        let z_row: Array1<f64> = self.row_index.iter().map(|i| z[*i]).collect();
        let z_col: Array1<f64> = self.col_index.iter().map(|i| z[*i]).collect();
        (&w * (&z_row - &z_col).mapv(|a| a.powi(2))).sum()
    }
}