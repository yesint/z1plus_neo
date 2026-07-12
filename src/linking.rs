//! Gauss linking number between chains — a topological invariant used to
//! *detect* whether the minimizer ever lets two chains pass through each other.
//!
//! For a pair of polylines the linking number is the double sum over segment
//! pairs of the signed solid angle (Klenin–Langowski formula). For open chains
//! it is not an integer, but it can only change discontinuously when a real
//! crossing occurs — so comparing the summed |Lk| before and after minimization
//! flags topology violations.

use molar::prelude::*;
use std::f64::consts::PI;

/// Sum of |Lk| over all unordered chain pairs (obstacle chains min-imaged to
/// the primary chain). Chains shorter than 2 nodes are skipped.
pub fn total_abs_linking(chains: &[Vec<Vector3f>], pbox: Option<&PeriodicBox>) -> f64 {
    let mut total = 0.0;
    for a in 0..chains.len() {
        if chains[a].len() < 2 {
            continue;
        }
        for b in (a + 1)..chains.len() {
            if chains[b].len() < 2 {
                continue;
            }
            total += pair_linking(&chains[a], &chains[b], pbox).abs();
        }
    }
    total
}

fn pair_linking(a: &[Vector3f], b: &[Vector3f], pbox: Option<&PeriodicBox>) -> f64 {
    // Min-image b to a's centroid so the pair is in one periodic image.
    let ca = centroid(a);
    let shifted: Vec<Vector3f> = match pbox {
        None => b.to_vec(),
        Some(p) => {
            let cb = centroid(b);
            let shift = p.shortest_vector(&(cb - ca)) - (cb - ca);
            b.iter().map(|x| x + shift).collect()
        }
    };

    let mut lk = 0.0;
    for i in 0..a.len() - 1 {
        for j in 0..shifted.len() - 1 {
            lk += segment_pair_solid_angle(&a[i], &a[i + 1], &shifted[j], &shifted[j + 1]);
        }
    }
    lk / (4.0 * PI)
}

fn centroid(c: &[Vector3f]) -> Vector3f {
    let mut s = Vector3f::zeros();
    for p in c {
        s += p;
    }
    s / (c.len() as Float)
}

/// Signed solid angle contribution of segments (r1,r2) and (r3,r4)
/// (Klenin–Langowski). Returns 0 for degenerate/parallel configurations.
fn segment_pair_solid_angle(r1: &Vector3f, r2: &Vector3f, r3: &Vector3f, r4: &Vector3f) -> f64 {
    let r1 = r1.cast::<f64>();
    let r2 = r2.cast::<f64>();
    let r3 = r3.cast::<f64>();
    let r4 = r4.cast::<f64>();

    let r13 = r3 - r1;
    let r14 = r4 - r1;
    let r24 = r4 - r2;
    let r23 = r3 - r2;

    let n1 = r13.cross(&r14);
    let n2 = r14.cross(&r24);
    let n3 = r24.cross(&r23);
    let n4 = r23.cross(&r13);

    let unit = |v: nalgebra::Vector3<f64>| {
        let n = v.norm();
        if n < 1e-12 {
            None
        } else {
            Some(v / n)
        }
    };
    let (n1, n2, n3, n4) = match (unit(n1), unit(n2), unit(n3), unit(n4)) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => return 0.0,
    };

    let clamp = |x: f64| x.clamp(-1.0, 1.0);
    let omega = clamp(n1.dot(&n2)).asin()
        + clamp(n2.dot(&n3)).asin()
        + clamp(n3.dot(&n4)).asin()
        + clamp(n4.dot(&n1)).asin();

    let r34 = r4 - r3;
    let r12 = r2 - r1;
    let sign = r34.cross(&r12).dot(&r13).signum();
    omega * sign
}
