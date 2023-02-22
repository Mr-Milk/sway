use ndarray::{ArrayBase, DataMut, Ix1};
use rand::Rng;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

pub fn default_arg<T>(arg: Option<T>, default_value: T) -> T {
    match arg {
        Some(data) => data,
        None => default_value,
    }
}

pub fn zscore_p(z: f64, two_tailed: bool) -> f64 {
    let norm_dist: Normal = Normal::new(0.0, 1.0).unwrap(); // follow the scipy's default
    let mut p: f64 = if z > 0.0 {
        1.0 - norm_dist.cdf(z)
    } else {
        norm_dist.cdf(z)
    };

    if two_tailed {
        p *= 2.0
    }

    p
}

pub fn chi2_p(chi2_value: f64, ddof: f64) -> f64 {
    let chi2_dist: ChiSquared = ChiSquared::new(ddof).unwrap();
    1.0 - chi2_dist.cdf(chi2_value)
}

// implement shuffle for ndarray
// borrow from rand crate

#[inline]
fn gen_index<R: Rng + ?Sized>(rng: &mut R, ubound: usize) -> usize {
    if ubound <= (u32::MAX as usize) {
        rng.gen_range(0..ubound as u32) as usize
    } else {
        rng.gen_range(0..ubound)
    }
}

pub trait Shuffle<A, S>
where
    S: DataMut<Elem = A>
{
    fn shuffle<R>(&mut self, rng: &mut R)
    where R: Rng + ?Sized;
}

impl<A, S> Shuffle<A, S> for ArrayBase<S, Ix1>
where
    S: DataMut<Elem = A>,
{
    fn shuffle<R>(&mut self, rng: &mut R) where R: Rng + ?Sized {
        for i in (1..self.len()).rev() {
            // invariant: elements with index > i have been locked in place.
            self.swap(i, gen_index(rng, i + 1));
        }
    }
}