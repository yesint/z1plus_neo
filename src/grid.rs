//! Uniform spatial hash of chain segments for fast neighbor queries.
//!
//! Each segment is filed under the cell of its (periodically wrapped) midpoint.
//! Because the minimizer caps segment length at `lmax`, a query for segments
//! near a small triangle only needs to look at cells within a few `lmax` of the
//! query point. Rebuilt once per sweep; queries return candidate `(chain, seg)`
//! identities which the caller re-tests against current coordinates.

use std::collections::HashMap;

use molar::prelude::*;

pub struct SegmentGrid {
    ncell: [i32; 3],
    cell: [Float; 3],
    ext: Vector3f,
    periodic: bool,
    origin: Vector3f,
    map: HashMap<(i32, i32, i32), Vec<(u32, u32)>>,
}

impl SegmentGrid {
    /// Build a grid over the current segments. `cell_target` is the desired
    /// cell size (use ~`lmax`).
    pub fn build(chains: &[Vec<Vector3f>], pbox: Option<&PeriodicBox>, cell_target: Float) -> Self {
        let (origin, ext, periodic) = match pbox {
            Some(b) => (Vector3f::zeros(), b.get_box_extents(), true),
            None => {
                // Bounding box of all nodes.
                let mut lo = Vector3f::repeat(Float::INFINITY);
                let mut hi = Vector3f::repeat(Float::NEG_INFINITY);
                for c in chains {
                    for p in c {
                        lo = lo.inf(p);
                        hi = hi.sup(p);
                    }
                }
                if !lo.x.is_finite() {
                    lo = Vector3f::zeros();
                    hi = Vector3f::repeat(1.0);
                }
                (lo, hi - lo, false)
            }
        };

        let ct = cell_target.max(1.0e-3);
        let mut ncell = [1i32; 3];
        let mut cell = [1.0 as Float; 3];
        for d in 0..3 {
            let n = (ext[d] / ct).floor().max(1.0);
            ncell[d] = n as i32;
            cell[d] = ext[d] / n;
            if !cell[d].is_finite() || cell[d] <= 0.0 {
                cell[d] = ct;
                ncell[d] = 1;
            }
        }

        let mut map: HashMap<(i32, i32, i32), Vec<(u32, u32)>> = HashMap::new();
        for (ci, c) in chains.iter().enumerate() {
            for k in 0..c.len().saturating_sub(1) {
                let mid = (c[k] + c[k + 1]) * 0.5;
                let key = Self::cell_of(&mid, &origin, &ext, &cell, &ncell, periodic);
                map.entry(key).or_default().push((ci as u32, k as u32));
            }
        }

        Self { ncell, cell, ext, periodic, origin, map }
    }

    #[inline]
    fn cell_of(
        p: &Vector3f,
        origin: &Vector3f,
        ext: &Vector3f,
        cell: &[Float; 3],
        ncell: &[i32; 3],
        periodic: bool,
    ) -> (i32, i32, i32) {
        let mut idx = [0i32; 3];
        for d in 0..3 {
            let mut x = p[d] - origin[d];
            if periodic {
                x -= (x / ext[d]).floor() * ext[d];
            }
            let mut c = (x / cell[d]).floor() as i32;
            if periodic {
                c = c.rem_euclid(ncell[d]);
            } else {
                c = c.clamp(0, ncell[d] - 1);
            }
            idx[d] = c;
        }
        (idx[0], idx[1], idx[2])
    }

    /// Collect candidate `(chain, seg)` whose midpoint cell is within `radius`
    /// of `center`. Candidates may repeat only across periodic wraps; callers
    /// re-test geometry so a stray duplicate is harmless.
    pub fn query(&self, center: &Vector3f, radius: Float, out: &mut Vec<(u32, u32)>) {
        out.clear();
        let base = Self::cell_of(center, &self.origin, &self.ext, &self.cell, &self.ncell, self.periodic);
        let mut r = [0i32; 3];
        for d in 0..3 {
            r[d] = (radius / self.cell[d]).ceil() as i32;
            // Never scan more than the whole box in a periodic dimension.
            if self.periodic {
                r[d] = r[d].min(self.ncell[d] / 2 + 1);
            }
        }
        for dx in -r[0]..=r[0] {
            for dy in -r[1]..=r[1] {
                for dz in -r[2]..=r[2] {
                    let key = self.wrap_key(base.0 + dx, base.1 + dy, base.2 + dz);
                    if let Some(v) = self.map.get(&key) {
                        out.extend_from_slice(v);
                    }
                }
            }
        }
    }

    #[inline]
    fn wrap_key(&self, i: i32, j: i32, k: i32) -> (i32, i32, i32) {
        if self.periodic {
            (
                i.rem_euclid(self.ncell[0]),
                j.rem_euclid(self.ncell[1]),
                k.rem_euclid(self.ncell[2]),
            )
        } else if i < 0
            || j < 0
            || k < 0
            || i >= self.ncell[0]
            || j >= self.ncell[1]
            || k >= self.ncell[2]
        {
            (i32::MIN, i32::MIN, i32::MIN) // guaranteed-empty key
        } else {
            (i, j, k)
        }
    }
}
