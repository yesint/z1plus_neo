//! Allen–Tildesley linked-cell list over the node pool.
//!
//! Cells bin the *folded* working coordinate `x`; obstacle queries gather the
//! non-empty bin heads over the `Mif` stencil around a point, so the per-node
//! obstacle scan is
//! O(neighbours) not O(N).
//!
//! Node membership is stored per id (`id_in_cell`, `nextbead`) parallel to the
//! `Pool`; `update` is a no-op when a moved node's cell is unchanged, else an
//! inline remove+add — the only cell call the sweep makes for a move.

use crate::pool::{ChainId, Id, Pool, NULL};

/// Orthorhombic periodic box (the benchmarks + the PE melt are all cubic; the
/// min-image is per-axis). Positions are kept folded to `[-L/2, L/2)`.
#[derive(Clone, Copy, Debug)]
pub struct BoxDims {
    pub l: [f64; 3],
}

impl BoxDims {
    #[inline]
    pub fn min_image(&self, mut d: [f64; 3]) -> [f64; 3] {
        for k in 0..3 {
            let l = self.l[k];
            if l > 0.0 {
                d[k] -= l * (d[k] / l).round();
            }
        }
        d
    }
    /// Fold a coordinate into the centred box `[-L/2, L/2)`.
    #[inline]
    pub fn fold(&self, mut p: [f64; 3]) -> [f64; 3] {
        for k in 0..3 {
            let l = self.l[k];
            if l > 0.0 {
                p[k] -= l * (p[k] / l).round();
            }
        }
        p
    }
    #[inline]
    pub fn dist2(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        let d = self.min_image([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    }
}

#[derive(Clone, Debug)]
pub struct Cells {
    pub bx: BoxDims,
    pub rcut: f64,
    /// cells per axis (>=1, clamped to upper_M_limit=300)
    pub m: [i32; 3],
    pub binsize: [f64; 3],
    /// = m/2 so that folded x in [-L/2,L/2) maps to cell in [0,m)
    pub xbshift: [f64; 3],
    /// neighbour stencil half-widths per axis: [0,0]|[0,1]|[-1,1]
    pub mif: [[i32; 2]; 3],
    /// head-of-bin id per linear cell (0 == empty)
    firstbead: Vec<Id>,
    /// next id in same bin (parallel to pool ids)
    nextbead: Vec<Id>,
    /// cached cell of each id (parallel to pool ids)
    id_in_cell: Vec<[i32; 3]>,
}

const UPPER_M_LIMIT: i32 = 300;

impl Cells {
    /// Build the grid for a box + search radius and bin all live pool nodes.
    pub fn build(pool: &Pool, bx: BoxDims, rcut: f64) -> Self {
        let mut m = [1i32; 3];
        let mut binsize = [0.0f64; 3];
        let mut xbshift = [0.0f64; 3];
        let mut mif = [[0i32; 2]; 3];
        for k in 0..3 {
            let mk = if rcut > 0.0 {
                ((bx.l[k] / rcut) as i32).clamp(1, UPPER_M_LIMIT)
            } else {
                1
            };
            m[k] = mk;
            binsize[k] = if mk > 0 { bx.l[k] / mk as f64 } else { bx.l[k] };
            xbshift[k] = mk as f64 / 2.0;
            mif[k] = match mk {
                1 => [0, 0],
                2 => [0, 1],
                _ => [-1, 1],
            };
        }
        let ncells = (m[0] * m[1] * m[2]) as usize;
        let cap = pool.capacity() + 1;
        let mut c = Cells {
            bx,
            rcut,
            m,
            binsize,
            xbshift,
            mif,
            firstbead: vec![NULL; ncells],
            nextbead: vec![NULL; cap],
            id_in_cell: vec![[0; 3]; cap],
        };
        c.rebuild(pool);
        c
    }

    #[inline]
    fn cell_of(&self, p: [f64; 3]) -> [i32; 3] {
        // Node positions are stored UNFOLDED (may lie outside [-L/2, L/2)); fold
        // into the central box before binning so a chain spanning several images
        // still maps to the correct (wrapped) cell. The sweep's geometry is
        // image-invariant, but the grid index must be computed on the folded
        // position or an out-of-box node would clamp to an edge cell and miss
        // its true periodic neighbours.
        let p = self.bx.fold(p);
        let mut c = [0i32; 3];
        for k in 0..3 {
            // folded x in [-L/2, L/2); + xbshift(=m/2) → [0, m); trunc then top-clamp
            let mut ci = (p[k] / self.binsize[k] + self.xbshift[k]).floor() as i32;
            if ci < 0 {
                ci = 0;
            }
            if ci >= self.m[k] {
                ci = self.m[k] - 1;
            }
            c[k] = ci;
        }
        c
    }

    #[inline]
    fn lin(&self, c: [i32; 3]) -> usize {
        ((c[0] * self.m[1] + c[1]) * self.m[2] + c[2]) as usize
    }

    /// Rebuild `firstbead` from scratch over all live nodes (used on setup and
    /// on adaptive rcut change). Resizes per-id arrays if the pool grew.
    pub fn rebuild(&mut self, pool: &Pool) {
        let cap = pool.capacity() + 1;
        if self.nextbead.len() < cap {
            self.nextbead.resize(cap, NULL);
            self.id_in_cell.resize(cap, [0; 3]);
        }
        for h in self.firstbead.iter_mut() {
            *h = NULL;
        }
        for c in 1..=pool.chains() as ChainId {
            for id in pool.walk(c) {
                self.add(pool, id);
            }
        }
    }

    /// Insert `id` at the head of its bin (O(1)).
    pub fn add(&mut self, pool: &Pool, id: Id) {
        let cell = self.cell_of(pool.x[id as usize]);
        self.id_in_cell[id as usize] = cell;
        let l = self.lin(cell);
        self.nextbead[id as usize] = self.firstbead[l];
        self.firstbead[l] = id;
    }

    /// Remove `id` from its cached bin (O(bin)).
    pub fn remove(&mut self, id: Id) {
        let cell = self.id_in_cell[id as usize];
        let l = self.lin(cell);
        let mut cur = self.firstbead[l];
        if cur == id {
            self.firstbead[l] = self.nextbead[id as usize];
            return;
        }
        while cur != NULL {
            let nx = self.nextbead[cur as usize];
            if nx == id {
                self.nextbead[cur as usize] = self.nextbead[id as usize];
                return;
            }
            cur = nx;
        }
    }

    /// Re-bin `id` after a move; no-op if its cell is unchanged.
    pub fn update(&mut self, pool: &Pool, id: Id) {
        let new_cell = self.cell_of(pool.x[id as usize]);
        if new_cell == self.id_in_cell[id as usize] {
            return;
        }
        self.remove(id);
        self.add(pool, id);
    }

    /// Call `f(id)` for every node in a bin around `p` (Mif stencil, toroidal
    /// wrap). May include the query node itself and duplicates only across
    /// distinct cells (never within a cell); callers filter by id.
    /// Cell coordinates of a position (public wrapper for stencil dedup).
    #[inline]
    pub fn cell_of_pos(&self, p: [f64; 3]) -> [i32; 3] {
        self.cell_of(p)
    }

    /// Cached cell coordinates of a live id.
    #[inline]
    pub fn cell_of_id(&self, id: Id) -> [i32; 3] {
        self.id_in_cell[id as usize]
    }

    /// Is cell `cc` inside the query stencil (Mif, toroidal) centred on `c0`?
    /// Used to deduplicate a segment reachable from both endpoints: if the other
    /// endpoint is itself in the stencil it will be visited as a candidate and
    /// emit the segment, so this endpoint can skip it without losing coverage.
    #[inline]
    pub fn in_stencil(&self, c0: [i32; 3], cc: [i32; 3]) -> bool {
        for k in 0..3 {
            let mut hit = false;
            let mut d = self.mif[k][0];
            while d <= self.mif[k][1] {
                if wrap(c0[k] + d, self.m[k]) == cc[k] {
                    hit = true;
                    break;
                }
                d += 1;
            }
            if !hit {
                return false;
            }
        }
        true
    }

    pub fn for_each_candidate<F: FnMut(Id)>(&self, p: [f64; 3], mut f: F) {
        let c0 = self.cell_of(p);
        for dx in self.mif[0][0]..=self.mif[0][1] {
            let ix = wrap(c0[0] + dx, self.m[0]);
            for dy in self.mif[1][0]..=self.mif[1][1] {
                let iy = wrap(c0[1] + dy, self.m[1]);
                for dz in self.mif[2][0]..=self.mif[2][1] {
                    let iz = wrap(c0[2] + dz, self.m[2]);
                    let l = self.lin([ix, iy, iz]);
                    let mut id = self.firstbead[l];
                    while id != NULL {
                        f(id);
                        id = self.nextbead[id as usize];
                    }
                }
            }
        }
    }
}

#[inline]
fn wrap(mut i: i32, m: i32) -> i32 {
    if m <= 0 {
        return 0;
    }
    while i < 0 {
        i += m;
    }
    while i >= m {
        i -= m;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_chain(n: usize, spacing: f64, origin: [f64; 3]) -> Vec<[f64; 3]> {
        (0..n)
            .map(|i| [origin[0] + i as f64 * spacing, origin[1], origin[2]])
            .collect()
    }

    /// The cell candidate set must be a SUPERSET of the brute-force set of all
    /// live nodes within rcut of the query point (consensus Phase-2 gate).
    #[test]
    fn candidate_set_superset_of_brute_within_rcut() {
        let l = 10.0;
        let bx = BoxDims { l: [l, l, l] };
        // several chains scattered so multiple cells are populated
        let chains = vec![
            grid_chain(8, 0.4, [-4.0, -3.0, 0.0]),
            grid_chain(8, 0.4, [1.0, 2.0, -1.0]),
            grid_chain(8, 0.4, [-2.0, 3.5, 2.0]),
            grid_chain(8, 0.4, [3.0, -3.5, -3.0]),
        ];
        // fold positions into the box first (as the sweep keeps x folded)
        let mut chains_folded = chains.clone();
        for c in chains_folded.iter_mut() {
            for p in c.iter_mut() {
                *p = bx.fold(*p);
            }
        }
        let mut pool = Pool::from_chains(&chains_folded);
        for id in 1..=pool.maxid_ever {
            pool.x[id as usize] = bx.fold(pool.x[id as usize]);
        }
        let rcut = 1.2;
        let cells = Cells::build(&pool, bx, rcut);

        // query at many points; candidate set must include every node within rcut
        let queries = [
            [0.0, 0.0, 0.0],
            [-4.0, -3.0, 0.0],
            [1.2, 2.1, -1.0],
            [4.9, 4.9, 4.9],
            [-4.9, -4.9, -4.9],
        ];
        for &q in &queries {
            let qf = bx.fold(q);
            let mut cand = std::collections::HashSet::new();
            cells.for_each_candidate(qf, |id| {
                cand.insert(id);
            });
            for c in 1..=pool.chains() as ChainId {
                for id in pool.walk(c) {
                    if bx.dist2(qf, pool.x[id as usize]) <= rcut * rcut {
                        assert!(
                            cand.contains(&id),
                            "node {id} within rcut of {qf:?} missing from candidates"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn update_is_noop_when_cell_unchanged_and_tracks_moves() {
        let l = 20.0;
        let bx = BoxDims { l: [l, l, l] };
        let chains = vec![grid_chain(10, 0.5, [0.0, 0.0, 0.0])];
        let mut pool = Pool::from_chains(&chains);
        for id in 1..=pool.maxid_ever {
            pool.x[id as usize] = bx.fold(pool.x[id as usize]);
        }
        let mut cells = Cells::build(&pool, bx, 1.0);
        let id = pool.idstart[1] + 3;
        let before = cells.id_in_cell[id as usize];
        // tiny move within the same cell → cached cell unchanged, membership intact
        pool.x[id as usize][0] += 1e-6;
        cells.update(&pool, id);
        assert_eq!(cells.id_in_cell[id as usize], before);
        // big move to a far cell → must be found near its new location, not old
        pool.x[id as usize] = bx.fold([7.3, -6.1, 4.2]);
        cells.update(&pool, id);
        let mut found_new = false;
        cells.for_each_candidate(pool.x[id as usize], |q| {
            if q == id {
                found_new = true;
            }
        });
        assert!(found_new, "moved node not found at its new cell");
        let mut found_old = false;
        cells.for_each_candidate(bx.fold([0.0 + 3.0 * 0.5, 0.0, 0.0]), |q| {
            if q == id {
                found_old = true;
            }
        });
        assert!(!found_old, "moved node still present at old cell");
    }

    #[test]
    fn single_cell_when_rcut_exceeds_box() {
        let l = 5.0;
        let bx = BoxDims { l: [l, l, l] };
        let chains = vec![grid_chain(6, 0.5, [-1.0, 0.0, 0.0])];
        let mut pool = Pool::from_chains(&chains);
        for id in 1..=pool.maxid_ever {
            pool.x[id as usize] = bx.fold(pool.x[id as usize]);
        }
        let cells = Cells::build(&pool, bx, 100.0);
        assert_eq!(cells.m, [1, 1, 1]);
        // every node is a candidate from anywhere
        let mut cand = std::collections::HashSet::new();
        cells.for_each_candidate([0.0, 0.0, 0.0], |id| {
            cand.insert(id);
        });
        assert_eq!(cand.len() as u32, pool.system_n);
    }
}
