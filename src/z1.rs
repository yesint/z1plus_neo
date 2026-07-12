//! The geometric shortest-multiple-disconnected-path minimizer.
//!
//! Algorithm (per frame): every chain is a polyline of nodes with fixed
//! endpoints. Each interior node `i` is treated in turn:
//!
//! * **Remove** it if the triangle `(i-1, i, i+1)` is pierced by no other
//!   chain — the straightening move `(i-1,i)+(i,i+1) -> (i-1,i+1)` sweeps that
//!   triangle, so an empty triangle means the collapse crosses nothing.
//! * Otherwise **slide** it toward the chord `[i-1, i+1]` (which shortens the
//!   contour) as far as a swept-triangle collision test permits, leaving it
//!   taut against the obstacle that blocks it.
//!
//! Both moves strictly shorten the contour and never let a chain pass through
//! another, so the process is monotone and converges to the shortest path
//! under the fixed topology. Surviving bent interior nodes are the kinks:
//! their count per chain is `Z`, the summed segment length is `Lpp`.

use molar::prelude::*;

use crate::geom::{
    aabb_overlap, closest_point_on_segment, seg_aabb_pair, segment_pierces_triangle, tri_aabb,
};
use crate::grid::SegmentGrid;

/// Chains with fewer than this many beads are "dumbbells": immobile obstacles
/// that constrain other chains but carry no entanglement statistics of their
/// own (matches Z1's treatment of `N_j <= 2`).
pub const MIN_TRUE_CHAIN: usize = 3;

const LEN_TOL: Float = 1.0e-4; // relative total-length change for convergence
const MOVE_EPS: Float = 1.0e-5; // minimum node displacement counted as "moved"
const BISECT_ITERS: usize = 24; // ~1e-7 resolution in the move fraction

/// Per-chain result for one frame.
#[derive(Clone, Debug)]
pub struct ChainResult {
    pub n_beads: usize,
    pub is_true: bool,
    pub z: usize,
    pub lpp: Float,
    pub ree: Float,
}

/// Result for one analyzed frame.
#[derive(Clone, Debug)]
pub struct FrameResult {
    pub chains: Vec<ChainResult>,
}

/// Options controlling the minimization.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Also forbid a chain from crossing itself (self-entanglements).
    pub self_entanglement: bool,
    /// Safety cap on the number of full sweeps.
    pub max_sweeps: usize,
    /// A node counts as a kink if its turning angle exceeds this (radians).
    pub kink_angle: Float,
    /// Maximum segment length. A node is removed only if the merged segment
    /// stays below this, so collinear "ghost" nodes remain on long straight
    /// stretches. Keeps triangles small enough that the minimum-image
    /// collision test is unambiguous under PBC. Set per-frame from the box.
    pub lmax: Float,
    /// Enable continuous node sliding (tightening). If false, only node
    /// removal is performed (topology-safe but leaves an over-long path).
    pub slide: bool,
    /// Use the spatial grid for neighbor queries. If false, brute-force scan
    /// (diagnostic — should give identical results, just slower).
    pub use_grid: bool,
    /// Enable node removal. If false, only sliding is performed (diagnostic).
    pub remove: bool,
    /// Chain thickness: the shortest path is kept at least this far from other
    /// chains, so nodes never sit exactly on an obstacle (which creates
    /// degenerate configurations that leak). Analogous to Z1's `thickness`.
    pub thickness: Float,
    /// Tightening target: if true, displace a blocked node toward the chord
    /// foot (Z1-style maximal length reduction, stopping at the first
    /// obstacle); if false, jump to the binding obstacle's wrap point.
    pub chord_slide: bool,
    /// Use a STRICT pierce test for node removal (remove glancing nodes that
    /// only come within `thickness`); leak-safe because contacts rest a
    /// `thickness` gap off obstacles. If false, removal is thickness-fat.
    pub strict_removal: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            self_entanglement: false,
            max_sweeps: 2000,
            kink_angle: 0.05, // ~2.9 degrees
            lmax: Float::INFINITY,
            slide: true,
            use_grid: true,
            remove: true,
            thickness: 0.01,
            chord_slide: false,
            // Fat removal is the only leak-safe option: it never lets a
            // chain cross another. Strict/edge removal tighten more but leak on
            // long coiled chains (Lpp dips below the true minimum) — the exact
            // safe-and-tight behaviour needs Z1's continuous-contact algorithm.
            strict_removal: false,
        }
    }
}

/// Analyze one frame: minimize a copy of `chains` and collect statistics.
pub fn analyze_frame(
    chains: &[Vec<Vector3f>],
    pbox: Option<&PeriodicBox>,
    opts: Options,
) -> FrameResult {
    let n_beads: Vec<usize> = chains.iter().map(|c| c.len()).collect();
    let ree: Vec<Float> = chains
        .iter()
        .map(|c| {
            if c.len() >= 2 {
                (c[c.len() - 1] - c[0]).norm()
            } else {
                0.0
            }
        })
        .collect();

    // Cap segment length so triangles stay small vs the box: keeps the
    // min-image collision test unambiguous AND lets the spatial grid localize
    // queries (many short cells). Small relative to the box, but never below a
    // few initial bond lengths so the chain is representable.
    let mut opts = opts;
    if opts.lmax.is_infinite() {
        if pbox.is_some() {
            let mut maxbond: Float = 0.0;
            for c in chains {
                for w in c.windows(2) {
                    maxbond = maxbond.max((w[1] - w[0]).norm());
                }
            }
            // Z1+'s lmax_factor defaults to 1, and the base lmax is the
            // maximum initial bond length over all chains (including rigid
            // dumbbell obstacles). It is not a fraction of the box size.
            opts.lmax = maxbond;
        }
    }

    let mut work = introduce_nodes_to_reduce_cost(chains, pbox);
    minimize(&mut work, pbox, opts);

    let results = (0..work.len())
        .map(|i| {
            let c = &work[i];
            let is_true = n_beads[i] >= MIN_TRUE_CHAIN;
            let lpp: Float = c.windows(2).map(|w| (w[1] - w[0]).norm()).sum();
            // Z = number of interior nodes that are genuine entanglement
            // contacts, i.e. whose removal would force a chain crossing.
            // Parameter-free (no angle threshold): a free bend can be removed
            // and does not count. Fall back to angle if requested via a
            // non-zero angle on an empty-neighborhood node.
            let z = if is_true {
                count_contacts(&work, i, pbox, opts)
            } else {
                0
            };
            ChainResult {
                n_beads: n_beads[i],
                is_true,
                z,
                lpp,
                ree: ree[i],
            }
        })
        .collect();

    FrameResult { chains: results }
}

/// Count interior nodes of chain `ci` that are genuine entanglement contacts:
/// a node is a kink iff collapsing it (triangle `p,a,q`) would be blocked by a
/// strict crossing of another chain. Free bends (removable) do not count.
fn count_contacts(
    chains: &[Vec<Vector3f>],
    ci: usize,
    pbox: Option<&PeriodicBox>,
    opts: Options,
) -> usize {
    let c = &chains[ci];
    if c.len() < 3 {
        return 0;
    }
    // Strict test: thickness=0, so only true topological contacts count.
    let strict = Options {
        thickness: 0.0,
        ..opts
    };
    let cos_thresh = opts.kink_angle.cos();
    let mut z = 0;
    for i in 1..c.len() - 1 {
        let (p, a, q) = (c[i - 1], c[i], c[i + 1]);
        // A genuine kink must be both a topological contact (its collapse is
        // blocked) AND actually bent — on a slightly-loose path a straight-
        // through node can still clip an obstacle at a glancing angle; that is
        // not an entanglement.
        let u = a - p;
        let v = q - a;
        let (nu, nv) = (u.norm(), v.norm());
        let bent = nu >= MOVE_EPS
            && nv >= MOVE_EPS
            && (u.dot(&v) / (nu * nv)).clamp(-1.0, 1.0) < cos_thresh;
        if bent && triangle_blocked(chains, ci, i, &p, &a, &q, pbox, strict, None) {
            z += 1;
        }
    }
    z
}

/// Minimize a copy of `chains` and return the resulting shortest paths
/// (diagnostic entry point, e.g. for the linking-number check).
pub fn minimize_chains(
    chains: &[Vec<Vector3f>],
    pbox: Option<&PeriodicBox>,
    opts: Options,
) -> Vec<Vec<Vector3f>> {
    let mut opts = opts;
    if opts.lmax.is_infinite() {
        if pbox.is_some() {
            let mut maxbond: Float = 0.0;
            for c in chains {
                for w in c.windows(2) {
                    maxbond = maxbond.max((w[1] - w[0]).norm());
                }
            }
            opts.lmax = maxbond;
        }
    }
    let mut work = introduce_nodes_to_reduce_cost(chains, pbox);
    minimize(&mut work, pbox, opts);
    work
}

/// Reproduce Z1+'s initial cost-reduction pass. True-chain bonds are bisected
/// until they are no longer than
///
/// `upper_bondl = min(mean_true_chain_bond_length, number_density^(-1/3))`.
///
/// Dumbbells remain two-node rigid obstacles. The inserted points are temporary
/// ghost nodes; later minimization/finalization may remove them.
fn introduce_nodes_to_reduce_cost(
    chains: &[Vec<Vector3f>],
    pbox: Option<&PeriodicBox>,
) -> Vec<Vec<Vector3f>> {
    let Some(pbox) = pbox else {
        return chains.to_vec();
    };

    let mut bond_sum = 0.0;
    let mut bond_count = 0usize;
    for chain in chains.iter().filter(|chain| chain.len() >= MIN_TRUE_CHAIN) {
        for segment in chain.windows(2) {
            bond_sum += (segment[1] - segment[0]).norm();
            bond_count += 1;
        }
    }
    if bond_count == 0 {
        return chains.to_vec();
    }

    let volume = pbox.get_matrix().determinant().abs();
    if !volume.is_finite() || volume <= 0.0 {
        return chains.to_vec();
    }
    let number_density = chains.iter().map(Vec::len).sum::<usize>() as Float / volume;
    let mean_bond = bond_sum / bond_count as Float;
    let upper_bond = mean_bond.min(number_density.powf(-1.0 / 3.0));
    if !upper_bond.is_finite() || upper_bond <= 0.0 {
        return chains.to_vec();
    }

    chains
        .iter()
        .map(|chain| {
            if chain.len() < MIN_TRUE_CHAIN {
                return chain.clone();
            }
            let mut refined = Vec::with_capacity(chain.len());
            refined.push(chain[0]);
            for segment in chain.windows(2) {
                let length = (segment[1] - segment[0]).norm();
                let mut pieces = 1usize;
                while length / pieces as Float > upper_bond {
                    pieces *= 2;
                }
                for piece in 1..=pieces {
                    let t = piece as Float / pieces as Float;
                    refined.push(segment[0] + (segment[1] - segment[0]) * t);
                }
            }
            refined
        })
        .collect()
}

fn total_len(chains: &[Vec<Vector3f>]) -> Float {
    chains.iter().map(|c| chain_len(c)).sum()
}

#[inline]
fn chain_len(c: &[Vector3f]) -> Float {
    c.windows(2).map(|w| (w[1] - w[0]).norm()).sum()
}

/// Iteratively straighten/tighten all chains until convergence.
fn minimize(chains: &mut [Vec<Vector3f>], pbox: Option<&PeriodicBox>, opts: Options) {
    let mut prev_len = total_len(chains);
    for sweep in 0..opts.max_sweeps {
        // Spatial index of segments, rebuilt each sweep. Only when lmax is
        // finite (a box is present); otherwise fall back to the brute scan.
        let grid = if opts.lmax.is_finite() && opts.use_grid {
            Some(SegmentGrid::build(chains, pbox, opts.lmax))
        } else {
            None
        };
        // Forward traversal. (Alternating the direction each sweep was tried
        // and is NOT a robust improvement: it converged to a different, often
        // looser local minimum — worse on benchmark-07. The `reverse` path is
        // kept for experimentation via a future flag.)
        let _ = sweep;
        let reverse = false;
        let mut changed = false;
        for ci in 0..chains.len() {
            changed |= process_chain(chains, ci, pbox, opts, grid.as_ref(), reverse);
        }
        let len = total_len(chains);
        if !changed {
            break;
        }
        if prev_len > 0.0 && (prev_len - len).abs() / prev_len < LEN_TOL {
            break;
        }
        prev_len = len;
    }
}

/// One pass over a single chain: remove or slide each interior node.
/// `reverse` traverses interior nodes from the far end (for bidirectional
/// sweeps).
fn process_chain(
    chains: &mut [Vec<Vector3f>],
    ci: usize,
    pbox: Option<&PeriodicBox>,
    opts: Options,
    grid: Option<&SegmentGrid>,
    reverse: bool,
) -> bool {
    if chains[ci].len() < MIN_TRUE_CHAIN {
        return false;
    }
    let mut changed = false;
    let mut i = if reverse { chains[ci].len() - 2 } else { 1 };
    while i >= 1 && i + 1 < chains[ci].len() {
        let p = chains[ci][i - 1];
        let a = chains[ci][i];
        let q = chains[ci][i + 1];

        // Removal: is the collapse triangle (p, a, q) free of other chains?
        // Only if the merged segment stays short (ghost-node cap), so triangles
        // never grow large enough to make the min-image test ambiguous.
        // Removal is allowed iff the collapse triangle (p,a,q) is not strictly
        // pierced (no crossing) AND the resulting edge p-q keeps a `thickness`
        // gap from all obstacles. This removes glancing nodes — where an
        // obstacle is near the node `a` but far from the new edge p-q — while
        // staying leak-safe: it never creates an edge that hugs an obstacle
        // (which would reintroduce the coincident-contact leak). With
        // `strict_removal=false`, fall back to the thickness-fat triangle test.
        let blocked = if opts.strict_removal {
            let strict = Options {
                thickness: 0.0,
                ..opts
            };
            triangle_blocked(chains, ci, i, &p, &a, &q, pbox, strict, grid)
                || edge_too_close(chains, ci, i, &p, &q, pbox, opts, grid)
        } else {
            triangle_too_close(chains, ci, i, &p, &a, &q, pbox, opts, grid)
        };
        if opts.remove && (p - q).norm() <= opts.lmax && !blocked {
            if VERIFY.with(|v| v.get()) {
                verify_removal(chains, ci, i, &p, &a, &q, pbox, opts);
            }
            chains[ci].remove(i);
            changed = true;
            // Re-examine the previous node, whose triangle just changed.
            if i > 1 {
                i -= 1;
            } else if reverse {
                break;
            }
            continue;
        }

        // Otherwise tighten. The node is blocked from collapsing by some
        // obstacle; move it toward the taut *contact point* on the binding
        // obstacle (giving a sharp kink), or, if none is found, toward the
        // perpendicular foot of the chord. Clip by the swept-safety check so
        // the move itself never crosses another chain.
        if !opts.slide {
            if reverse {
                if i <= 1 {
                    break;
                }
                i -= 1;
            } else {
                i += 1;
            }
            continue;
        }
        let target = if opts.chord_slide {
            closest_point_on_segment(&a, &p, &q)
        } else {
            binding_contact(chains, ci, i, &p, &a, &q, pbox, opts, grid)
                .unwrap_or_else(|| closest_point_on_segment(&a, &p, &q))
        };
        let tmax = max_safe_t(chains, ci, i, &p, &q, &a, &target, pbox, opts, grid);
        if tmax > 0.0 {
            let newpos = a + (target - a) * tmax;
            if (newpos - a).norm() > MOVE_EPS {
                if VERIFY.with(|v| v.get()) {
                    verify_move(chains, ci, i, &p, &q, &a, &newpos, pbox, opts);
                }
                chains[ci][i] = newpos;
                changed = true;
            }
        }
        if reverse {
            if i <= 1 {
                break;
            }
            i -= 1;
        } else {
            i += 1;
        }
    }
    changed
}

/// Largest fraction `t in [0,1]` such that moving node `i` from `a` to
/// `a + t*(target-a)` sweeps no other chain (monotone: a larger sweep triangle
/// contains a smaller one, so this is a clean bisection).
#[allow(clippy::too_many_arguments)]
fn max_safe_t(
    chains: &[Vec<Vector3f>],
    ci: usize,
    i: usize,
    p: &Vector3f,
    q: &Vector3f,
    a: &Vector3f,
    target: &Vector3f,
    pbox: Option<&PeriodicBox>,
    opts: Options,
    grid: Option<&SegmentGrid>,
) -> Float {
    if swept_clear(chains, ci, i, p, q, a, target, 1.0, pbox, opts, grid) {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0_f32 as Float, 1.0_f32 as Float);
    for _ in 0..BISECT_ITERS {
        let mid = 0.5 * (lo + hi);
        if swept_clear(chains, ci, i, p, q, a, target, mid, pbox, opts, grid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Are both swept triangles `(p, a, M)` and `(q, a, M)` (with `M = a + t*(target-a)`)
/// free of other chains?
#[allow(clippy::too_many_arguments)]
fn swept_clear(
    chains: &[Vec<Vector3f>],
    ci: usize,
    i: usize,
    p: &Vector3f,
    q: &Vector3f,
    a: &Vector3f,
    target: &Vector3f,
    t: Float,
    pbox: Option<&PeriodicBox>,
    opts: Options,
    grid: Option<&SegmentGrid>,
) -> bool {
    let m = a + (target - a) * t;
    !triangle_blocked(chains, ci, i, p, a, &m, pbox, opts, grid)
        && !triangle_blocked(chains, ci, i, q, a, &m, pbox, opts, grid)
}

/// Is triangle `(t0,t1,t2)` pierced by a segment of another chain (or, if
/// `self_entanglement`, a non-adjacent segment of the same chain)?
#[allow(clippy::too_many_arguments)]
fn triangle_blocked(
    chains: &[Vec<Vector3f>],
    ci: usize,
    i: usize,
    t0: &Vector3f,
    t1: &Vector3f,
    t2: &Vector3f,
    pbox: Option<&PeriodicBox>,
    opts: Options,
    grid: Option<&SegmentGrid>,
) -> bool {
    let centroid = (t0 + t1 + t2) / 3.0;
    let (tlo, thi) = tri_aabb(t0, t1, t2);
    let margin = (t0 - t1).norm().max((t1 - t2).norm()).max((t2 - t0).norm());

    // Test one candidate segment (cj, k); returns true if it pierces.
    let hits = |cj: usize, k: usize| -> bool {
        let same = cj == ci;
        if same && !opts.self_entanglement {
            return false;
        }
        // Skip segments incident to nodes i-1, i, i+1 of the same chain.
        if same && k + 2 >= i && k <= i + 1 {
            return false;
        }
        let chain = &chains[cj];
        if k + 1 >= chain.len() {
            return false;
        }
        let (q0, q1) = image_segment(&chain[k], &chain[k + 1], &centroid, pbox);
        let (qlo, qhi) = seg_aabb_pair(&q0, &q1);
        if !aabb_overlap(&tlo, &thi, &qlo, &qhi, margin) {
            return false;
        }
        // Obstacles are "fat": block if the segment pierces the triangle or
        // comes within `thickness` of it. Keeps a consistent gap so a chain
        // cannot slide through the thin space between two other chains.
        if segment_pierces_triangle(&q0, &q1, t0, t1, t2) {
            return true;
        }
        seg_tri_min_dist(&q0, &q1, t0, t1, t2) < opts.thickness
    };

    match grid {
        Some(g) => {
            let mut cand: Vec<(u32, u32)> = Vec::new();
            g.query(&centroid, margin + 2.0 * opts.lmax, &mut cand);
            for (cj, k) in cand {
                if hits(cj as usize, k as usize) {
                    return true;
                }
            }
            false
        }
        None => {
            for cj in 0..chains.len() {
                for k in 0..chains[cj].len().saturating_sub(1) {
                    if hits(cj, k) {
                        return true;
                    }
                }
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vector3f {
        Vector3f::new(x as Float, y as Float, z as Float)
    }

    #[test]
    fn single_chain_straightens_fully() {
        // No obstacles: a kinked chain must collapse to the straight line.
        let chains = vec![vec![v(0., 0., 0.), v(1., 1., 0.), v(2., 0., 0.)]];
        let fr = analyze_frame(&chains, None, Options::default());
        let c = &fr.chains[0];
        assert!((c.lpp - 2.0).abs() < 1e-3, "lpp={} (expected 2.0)", c.lpp);
        assert_eq!(c.z, 0, "expected no kinks");
    }

    #[test]
    fn many_node_arc_over_bar_keeps_kink() {
        // 11-node semicircle (radius 5) from (-5,0,0) to (5,0,0). A rigid bar
        // sits along z at (0, 2.5). Combined remove+slide must tighten the arc
        // taut over the bar: Lpp = 2*sqrt(25+6.25) = 11.18, Z = 1 — not
        // straighten through it (Lpp -> 10, Z -> 0).
        let mut arc = Vec::new();
        for k in 0..11 {
            let theta = std::f64::consts::PI * (1.0 - k as f64 / 10.0);
            arc.push(v(5.0 * theta.cos() as f32, 5.0 * theta.sin() as f32, 0.0));
        }
        let bar = vec![v(0.0, 2.5, -3.0), v(0.0, 2.5, 3.0)];
        let opts = Options {
            lmax: 100.0,
            ..Options::default()
        };
        let fr = analyze_frame(&vec![arc, bar], None, opts);
        let c = &fr.chains[0];
        assert!(
            c.lpp > 10.8 && c.lpp < 11.6,
            "arc lpp={} (expected ~11.18)",
            c.lpp
        );
        assert_eq!(c.z, 1, "expected one kink over the bar, got z={}", c.z);
    }

    #[test]
    fn many_node_arc_over_bar_pbc_keeps_kink() {
        // The 11-node arc test, but periodic: the bar sits a full box away in z
        // (17..23, wrapping to -3..3) so it only blocks via the minimum image,
        // and combined remove+slide+contact must still keep the sharp kink.
        let mut mat = Matrix3f::zeros();
        mat[(0, 0)] = 20.0;
        mat[(1, 1)] = 20.0;
        mat[(2, 2)] = 20.0;
        let pbox = PeriodicBox::from_matrix(mat).unwrap();
        let mut arc = Vec::new();
        for k in 0..11 {
            let theta = std::f64::consts::PI * (1.0 - k as f64 / 10.0);
            arc.push(v(5.0 * theta.cos() as f32, 5.0 * theta.sin() as f32, 0.0));
        }
        let bar = vec![v(0.0, 2.5, 17.0), v(0.0, 2.5, 23.0)];
        let opts = Options {
            lmax: 100.0,
            ..Options::default()
        };
        let fr = analyze_frame(&vec![arc, bar], Some(&pbox), opts);
        let c = &fr.chains[0];
        assert!(
            c.lpp > 10.8 && c.lpp < 11.6,
            "pbc arc lpp={} (expected ~11.18)",
            c.lpp
        );
        assert_eq!(c.z, 1, "expected one kink, got z={}", c.z);
    }

    #[test]
    fn bar_obstacle_across_pbc_keeps_kink() {
        // Same as bar_obstacle_keeps_kink but the bar is placed a full box away
        // in z (8..12) so it only blocks via the minimum image (wraps to -2..2).
        let mut mat = Matrix3f::zeros();
        mat[(0, 0)] = 10.0;
        mat[(1, 1)] = 10.0;
        mat[(2, 2)] = 10.0;
        let pbox = PeriodicBox::from_matrix(mat).unwrap();
        let a = vec![v(-2., 0., 0.), v(0., 3., 0.), v(2., 0., 0.)];
        let bar = vec![v(0., 1.5, 8.), v(0., 1.5, 12.)];
        let chains = vec![a, bar];
        let opts = Options {
            lmax: 100.0,
            ..Options::default()
        };
        let fr = analyze_frame(&chains, Some(&pbox), opts);
        let ca = &fr.chains[0];
        assert!(
            ca.lpp > 4.8 && ca.lpp < 5.2,
            "pbc obstacle missed: lpp={}",
            ca.lpp
        );
        assert_eq!(ca.z, 1, "expected one kink across PBC, got z={}", ca.z);
    }

    #[test]
    fn bar_obstacle_keeps_kink() {
        // Chain A arcs up to y=3; a rigid bar (dumbbell) lies along z at (0,1.5).
        // Straightening A toward the x-axis must be blocked at y=1.5, leaving a
        // kink: Lpp = 2*sqrt(2^2 + 1.5^2) = 5.0, Z = 1.
        let a = vec![v(-2., 0., 0.), v(0., 3., 0.), v(2., 0., 0.)];
        let bar = vec![v(0., 1.5, -2.), v(0., 1.5, 2.)];
        let chains = vec![a, bar];
        let fr = analyze_frame(&chains, None, Options::default());
        let ca = &fr.chains[0];
        assert!(
            ca.lpp > 4.8 && ca.lpp < 5.2,
            "lpp={} (expected ~5.0)",
            ca.lpp
        );
        assert_eq!(ca.z, 1, "expected one kink, got z={}", ca.z);
    }
}

/// Is any obstacle within `opts.thickness` of triangle `(t0,t1,t2)`? Used for
/// the REMOVAL test: a node in contact with another chain (distance ~0) must
/// not be collapsed, else a real entanglement is destroyed. Distance-based so
/// it catches coincident/contact nodes that a pure pierce test misses at the
/// triangle vertices.
#[allow(clippy::too_many_arguments)]
fn triangle_too_close(
    chains: &[Vec<Vector3f>],
    ci: usize,
    i: usize,
    t0: &Vector3f,
    t1: &Vector3f,
    t2: &Vector3f,
    pbox: Option<&PeriodicBox>,
    opts: Options,
    grid: Option<&SegmentGrid>,
) -> bool {
    let centroid = (t0 + t1 + t2) / 3.0;
    let (tlo, thi) = tri_aabb(t0, t1, t2);
    let margin = (t0 - t1).norm().max((t1 - t2).norm()).max((t2 - t0).norm()) + opts.thickness;

    let check = |cj: usize, k: usize| -> bool {
        let same = cj == ci;
        if same && !opts.self_entanglement {
            return false;
        }
        if same && k + 2 >= i && k <= i + 1 {
            return false;
        }
        let chain = &chains[cj];
        if k + 1 >= chain.len() {
            return false;
        }
        let (s0, s1) = image_segment(&chain[k], &chain[k + 1], &centroid, pbox);
        let (qlo, qhi) = seg_aabb_pair(&s0, &s1);
        if !aabb_overlap(&tlo, &thi, &qlo, &qhi, margin) {
            return false;
        }
        if segment_pierces_triangle(&s0, &s1, t0, t1, t2) {
            return true;
        }
        seg_tri_min_dist(&s0, &s1, t0, t1, t2) < opts.thickness
    };

    match grid {
        Some(g) => {
            let mut cand = Vec::new();
            g.query(&centroid, margin + 2.0 * opts.lmax, &mut cand);
            cand.iter().any(|&(cj, k)| check(cj as usize, k as usize))
        }
        None => (0..chains.len())
            .any(|cj| (0..chains[cj].len().saturating_sub(1)).any(|k| check(cj, k))),
    }
}

/// Verify (independently) that collapsing node `i` — replacing `p-a-q` by
/// `p-q`, sweeping triangle `(p,a,q)` — crosses no obstacle. Prints the first
/// genuine (non-vertex-touch) intersection found.
#[allow(clippy::too_many_arguments)]
fn verify_removal(
    chains: &[Vec<Vector3f>],
    ci: usize,
    i: usize,
    p: &Vector3f,
    a: &Vector3f,
    q: &Vector3f,
    pbox: Option<&PeriodicBox>,
    opts: Options,
) {
    let centroid = (p + a + q) / 3.0;
    for (cj, chain) in chains.iter().enumerate() {
        if cj == ci && !opts.self_entanglement {
            continue;
        }
        for k in 0..chain.len().saturating_sub(1) {
            if cj == ci && k + 2 >= i && k <= i + 1 {
                continue;
            }
            let (s0, s1) = image_segment(&chain[k], &chain[k + 1], &centroid, pbox);
            // Ignore obstacle segments that share an endpoint with the triangle
            // (coincident contact) — those are legitimate, not a crossing.
            let touches = [p, a, q]
                .iter()
                .any(|v| (*v - s0).norm() < 1e-3 || (*v - s1).norm() < 1e-3);
            if touches {
                continue;
            }
            if parametric_seg_tri_intersect(&s0, &s1, p, a, q) {
                eprintln!(
                    "REMOVAL-CROSS chain {ci} node {i} vs chain {cj} seg {k}\n  P={p:?} A={a:?} Q={q:?}\n  S0={s0:?} S1={s1:?}"
                );
                return;
            }
        }
    }
}

/// Would the edge `p-q` (created by removing node `i`) come within `thickness`
/// of any obstacle segment? Used to keep node removal leak-safe: never create a
/// collapsed edge that hugs another chain.
#[allow(clippy::too_many_arguments)]
fn edge_too_close(
    chains: &[Vec<Vector3f>],
    ci: usize,
    i: usize,
    p: &Vector3f,
    q: &Vector3f,
    pbox: Option<&PeriodicBox>,
    opts: Options,
    grid: Option<&SegmentGrid>,
) -> bool {
    let mid = (p + q) * 0.5;
    let (elo, ehi) = seg_aabb_pair(p, q);
    let margin = (p - q).norm() * 0.5 + opts.thickness;

    let check = |cj: usize, k: usize| -> bool {
        let same = cj == ci;
        if same && !opts.self_entanglement {
            return false;
        }
        // Skip segments incident to nodes i-1, i, i+1 of the same chain.
        if same && k + 2 >= i && k <= i + 1 {
            return false;
        }
        let chain = &chains[cj];
        if k + 1 >= chain.len() {
            return false;
        }
        let (s0, s1) = image_segment(&chain[k], &chain[k + 1], &mid, pbox);
        let (qlo, qhi) = seg_aabb_pair(&s0, &s1);
        if !aabb_overlap(&elo, &ehi, &qlo, &qhi, margin) {
            return false;
        }
        seg_seg_min_dist(p, q, &s0, &s1) < opts.thickness
    };

    match grid {
        Some(g) => {
            let mut cand = Vec::new();
            g.query(&mid, margin + 2.0 * opts.lmax, &mut cand);
            cand.iter().any(|&(cj, k)| check(cj as usize, k as usize))
        }
        None => (0..chains.len())
            .any(|cj| (0..chains[cj].len().saturating_sub(1)).any(|k| check(cj, k))),
    }
}

/// Minimum distance between segment `[s0,s1]` and triangle `(t0,t1,t2)`.
/// Zero if they intersect.
fn seg_tri_min_dist(
    s0: &Vector3f,
    s1: &Vector3f,
    t0: &Vector3f,
    t1: &Vector3f,
    t2: &Vector3f,
) -> Float {
    if parametric_seg_tri_intersect(s0, s1, t0, t1, t2) {
        return 0.0;
    }
    let mut d = seg_seg_min_dist(s0, s1, t0, t1);
    d = d.min(seg_seg_min_dist(s0, s1, t1, t2));
    d = d.min(seg_seg_min_dist(s0, s1, t2, t0));
    d = d.min(point_tri_dist(s0, t0, t1, t2));
    d.min(point_tri_dist(s1, t0, t1, t2))
}

/// Distance from point `p` to triangle `(t0,t1,t2)`.
fn point_tri_dist(p: &Vector3f, t0: &Vector3f, t1: &Vector3f, t2: &Vector3f) -> Float {
    let e1 = t1 - t0;
    let e2 = t2 - t0;
    let n = e1.cross(&e2);
    let nn = n.norm();
    if nn < 1e-9 {
        return (p - t0).norm();
    }
    let nhat = n / nn;
    let dist_plane = (p - t0).dot(&nhat);
    let proj = p - nhat * dist_plane; // projection onto plane
                                      // Barycentric containment.
    let d00 = e1.dot(&e1);
    let d01 = e1.dot(&e2);
    let d11 = e2.dot(&e2);
    let xp = proj - t0;
    let d20 = xp.dot(&e1);
    let d21 = xp.dot(&e2);
    let det = d00 * d11 - d01 * d01;
    if det.abs() > 1e-12 {
        let v = (d11 * d20 - d01 * d21) / det;
        let w = (d00 * d21 - d01 * d20) / det;
        if v >= 0.0 && w >= 0.0 && v + w <= 1.0 {
            return dist_plane.abs();
        }
    }
    // Outside: nearest of the three edges.
    let mut d = seg_seg_min_dist(p, p, t0, t1);
    d = d.min(seg_seg_min_dist(p, p, t1, t2));
    d.min(seg_seg_min_dist(p, p, t2, t0))
}

/// Among obstacle segments that pierce the collapse triangle `(p, a, q)` (i.e.
/// block removal of node `a`), return the taut contact point that forces the
/// longest detour: `argmax over blocking S of min_{x in S} |p-x| + |x-q|`.
/// Returns `None` if nothing blocks (then the node is collapsible / free to
/// slide to the chord). The point is in `a`'s (min-imaged) frame.
#[allow(clippy::too_many_arguments)]
fn binding_contact(
    chains: &[Vec<Vector3f>],
    ci: usize,
    i: usize,
    p: &Vector3f,
    a: &Vector3f,
    q: &Vector3f,
    pbox: Option<&PeriodicBox>,
    opts: Options,
    grid: Option<&SegmentGrid>,
) -> Option<Vector3f> {
    let centroid = (p + a + q) / 3.0;
    let (tlo, thi) = tri_aabb(p, a, q);
    let margin = (p - a).norm().max((a - q).norm()).max((q - p).norm());

    let mut best: Option<(Float, Vector3f)> = None;
    let mut consider = |cj: usize, k: usize| {
        let same = cj == ci;
        if same && !opts.self_entanglement {
            return;
        }
        if same && k + 2 >= i && k <= i + 1 {
            return;
        }
        let chain = &chains[cj];
        if k + 1 >= chain.len() {
            return;
        }
        let (s0, s1) = image_segment(&chain[k], &chain[k + 1], &centroid, pbox);
        let (qlo, qhi) = seg_aabb_pair(&s0, &s1);
        if !aabb_overlap(&tlo, &thi, &qlo, &qhi, margin) {
            return;
        }
        if !segment_pierces_triangle(&s0, &s1, p, a, q) {
            return;
        }
        let x = wrap_point(&s0, &s1, p, q);
        // Keep the node a `thickness` gap OFF the obstacle so it never lands
        // exactly on it (coincident contacts are degenerate and leak). Offset
        // along the component of (a - x) perpendicular to the obstacle segment;
        // fall back to (a - x), then to the collapse-triangle normal.
        let ds = (s1 - s0).normalize();
        let ax = a - x;
        let perp = ax - ds * ax.dot(&ds);
        let off_dir = if perp.norm() > 1.0e-4 {
            perp.normalize()
        } else if ax.norm() > 1.0e-4 {
            ax.normalize()
        } else {
            let n = (a - p).cross(&(q - p));
            if n.norm() > 1.0e-6 {
                n.normalize()
            } else {
                ds
            }
        };
        let x = x + off_dir * opts.thickness;
        let detour = (p - x).norm() + (x - q).norm();
        if best.map_or(true, |(d, _)| detour > d) {
            best = Some((detour, x));
        }
    };

    match grid {
        Some(g) => {
            let mut cand = Vec::new();
            g.query(&centroid, margin + 2.0 * opts.lmax, &mut cand);
            for (cj, k) in cand {
                consider(cj as usize, k as usize);
            }
        }
        None => {
            for cj in 0..chains.len() {
                for k in 0..chains[cj].len().saturating_sub(1) {
                    consider(cj, k);
                }
            }
        }
    }
    best.map(|(_, x)| x)
}

/// Point `x` on segment `[s0,s1]` minimizing `|p-x| + |x-q|` (convex in the
/// segment parameter; ternary search).
fn wrap_point(s0: &Vector3f, s1: &Vector3f, p: &Vector3f, q: &Vector3f) -> Vector3f {
    let f = |u: Float| {
        let x = s0 + (s1 - s0) * u;
        (p - x).norm() + (x - q).norm()
    };
    let (mut lo, mut hi) = (0.0 as Float, 1.0 as Float);
    for _ in 0..40 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if f(m1) < f(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let u = 0.5 * (lo + hi);
    s0 + (s1 - s0) * u
}

thread_local! {
    /// Enable the independent move verifier (set from ENTANGL_VERIFY env).
    pub static VERIFY: std::cell::Cell<bool> = std::cell::Cell::new(false);
}

/// Independent check (segment–segment min distance sampled along the sweep,
/// no orient3d): if moving node `i` from `a` to `m` makes either of its
/// segments cross an obstacle — distance dips to ~0 in the *interior* of the
/// sweep then recovers — print the geometry. Catches false negatives in the
/// orient3d swept test.
#[allow(clippy::too_many_arguments)]
fn verify_move(
    chains: &[Vec<Vector3f>],
    ci: usize,
    i: usize,
    p: &Vector3f,
    q: &Vector3f,
    a: &Vector3f,
    m: &Vector3f,
    pbox: Option<&PeriodicBox>,
    opts: Options,
) {
    let centroid = (p + a + q + m) / 4.0;
    for (cj, chain) in chains.iter().enumerate() {
        if cj == ci && !opts.self_entanglement {
            continue;
        }
        for k in 0..chain.len().saturating_sub(1) {
            if cj == ci && k + 2 >= i && k <= i + 1 {
                continue;
            }
            let (s0, s1) = image_segment(&chain[k], &chain[k + 1], &centroid, pbox);
            // For each swept triangle of the applied move, cross-check the
            // production predicate against an INDEPENDENT parametric intersection.
            for &fixed in &[p, q] {
                let indep = parametric_seg_tri_intersect(&s0, &s1, fixed, a, m);
                let ours = segment_pierces_triangle(&s0, &s1, fixed, a, m);
                if indep && !ours {
                    eprintln!(
                        "PREDICATE-FN chain {ci} node {i} vs chain {cj} seg {k}: indep=true ours=false\n  fixed={fixed:?} A={a:?} M={m:?}\n  S0={s0:?} S1={s1:?}"
                    );
                    return;
                }
            }
        }
    }
}

/// Independent parametric segment/triangle intersection in f64 (plane
/// crossing + barycentric containment). Used only to cross-check the
/// production orient3d predicate in `verify_move`.
fn parametric_seg_tri_intersect(
    u0: &Vector3f,
    u1: &Vector3f,
    t0: &Vector3f,
    t1: &Vector3f,
    t2: &Vector3f,
) -> bool {
    let u0 = u0.cast::<f64>();
    let u1 = u1.cast::<f64>();
    let t0 = t0.cast::<f64>();
    let e1 = t1.cast::<f64>() - t0;
    let e2 = t2.cast::<f64>() - t0;
    let n = e1.cross(&e2);
    let dir = u1 - u0;
    let denom = n.dot(&dir);
    if denom.abs() < 1e-12 {
        return false; // parallel; ignore coplanar here (rare)
    }
    let s = n.dot(&(t0 - u0)) / denom;
    if !(0.0..=1.0).contains(&s) {
        return false; // crossing outside the segment
    }
    let x = u0 + dir * s; // intersection with plane
                          // Barycentric coordinates of x in the triangle.
    let d00 = e1.dot(&e1);
    let d01 = e1.dot(&e2);
    let d11 = e2.dot(&e2);
    let xp = x - t0;
    let d20 = xp.dot(&e1);
    let d21 = xp.dot(&e2);
    let det = d00 * d11 - d01 * d01;
    if det.abs() < 1e-18 {
        return false;
    }
    let v = (d11 * d20 - d01 * d21) / det;
    let w = (d00 * d21 - d01 * d20) / det;
    v >= 0.0 && w >= 0.0 && v + w <= 1.0
}

/// Minimum distance between segments `[p1,p2]` and `[p3,p4]` (standard).
fn seg_seg_min_dist(p1: &Vector3f, p2: &Vector3f, p3: &Vector3f, p4: &Vector3f) -> Float {
    const TINY: Float = 1.0e-9;
    let d1 = p2 - p1;
    let d2 = p4 - p3;
    let r = p1 - p3;
    let a = d1.dot(&d1);
    let e = d2.dot(&d2);
    let f = d2.dot(&r);
    let (mut s, mut t);
    if a <= TINY && e <= TINY {
        return r.norm();
    }
    if a <= TINY {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(&r);
        if e <= TINY {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(&d2);
            let denom = a * e - b * b;
            s = if denom.abs() > TINY {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    let c1 = p1 + d1 * s;
    let c2 = p3 + d2 * t;
    (c1 - c2).norm()
}

/// Translate segment `(s0,s1)` rigidly to the periodic image nearest `target`.
#[inline]
fn image_segment(
    s0: &Vector3f,
    s1: &Vector3f,
    target: &Vector3f,
    pbox: Option<&PeriodicBox>,
) -> (Vector3f, Vector3f) {
    match pbox {
        None => (*s0, *s1),
        Some(b) => {
            let mid = (s0 + s1) * 0.5;
            let shift = b.shortest_vector(&(mid - target)) - (mid - target);
            (s0 + shift, s1 + shift)
        }
    }
}
