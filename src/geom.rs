//! Low-level geometric predicates used by the SMDP minimizer.

use molar::prelude::*;

/// Numerical tolerance (in the coordinate units of the input, nm for GROMACS).
pub const EPS: Float = 1.0e-6;

/// Signed volume (×6) of the tetrahedron `(a,b,c,d)`, i.e. `(b-a)·((c-a)×(d-a))`,
/// evaluated in `f64`. Only the *sign* is used; computing in f64 keeps that sign
/// meaningful even when coordinates are stored in f32.
#[inline]
fn orient3d(a: &Vector3f, b: &Vector3f, c: &Vector3f, d: &Vector3f) -> f64 {
    let ba = (b - a).cast::<f64>();
    let ca = (c - a).cast::<f64>();
    let da = (d - a).cast::<f64>();
    ba.dot(&ca.cross(&da))
}

/// Does the segment `q0..q1` cross triangle `t0,t1,t2`?
///
/// Robust orient3d (signed-volume) formulation (Guigue–Devillers style): the
/// segment crosses iff its endpoints lie on opposite sides of the triangle
/// plane AND the segment passes inside all three edges. Division-free, so —
/// unlike Möller–Trumbore — it does not amplify round-off at grazing angles.
/// Boundary contacts (a zero volume) are treated inclusively, i.e. as a block:
/// over-blocking only leaves the path slightly long, whereas a missed crossing
/// lets chains pass through each other.
#[inline]
pub fn segment_pierces_triangle(
    q0: &Vector3f,
    q1: &Vector3f,
    t0: &Vector3f,
    t1: &Vector3f,
    t2: &Vector3f,
) -> bool {
    let a0 = orient3d(q0, t0, t1, t2);
    let a1 = orient3d(q1, t0, t1, t2);

    // Both endpoints strictly on the same side of the plane: cannot cross.
    if (a0 > 0.0 && a1 > 0.0) || (a0 < 0.0 && a1 < 0.0) {
        return false;
    }
    // Segment lies in the triangle plane: 2D overlap test.
    if a0 == 0.0 && a1 == 0.0 {
        return coplanar_overlap(q0, q1, t0, t1, t2, &(t1 - t0), &(t2 - t0));
    }

    // Straddles (or touches) the plane; require it to pass inside all edges.
    let b0 = orient3d(q0, q1, t0, t1);
    let b1 = orient3d(q0, q1, t1, t2);
    let b2 = orient3d(q0, q1, t2, t0);
    let nonneg = b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0;
    let nonpos = b0 <= 0.0 && b1 <= 0.0 && b2 <= 0.0;
    nonneg || nonpos
}

/// Coplanar fallback: is the segment `q0..q1` within the triangle plane and
/// overlapping the triangle (in 2D)?
fn coplanar_overlap(
    q0: &Vector3f,
    q1: &Vector3f,
    t0: &Vector3f,
    t1: &Vector3f,
    t2: &Vector3f,
    e1: &Vector3f,
    e2: &Vector3f,
) -> bool {
    let n = e1.cross(e2);
    let nn = n.norm();
    if nn < EPS {
        return false; // degenerate triangle
    }
    let nhat = n / nn;
    // Both segment endpoints must be close to the triangle plane.
    let d0 = (q0 - t0).dot(&nhat);
    let d1 = (q1 - t0).dot(&nhat);
    let plane_tol = 1.0e-3;
    if d0.abs() > plane_tol || d1.abs() > plane_tol {
        return false;
    }
    // Project to the 2D plane by dropping the largest-magnitude normal axis.
    let (i, j) = if n.x.abs() >= n.y.abs() && n.x.abs() >= n.z.abs() {
        (1, 2)
    } else if n.y.abs() >= n.z.abs() {
        (0, 2)
    } else {
        (0, 1)
    };
    let p2 = |v: &Vector3f| Vector2f::new(v[i], v[j]);
    seg_tri_overlap_2d(&p2(q0), &p2(q1), &p2(t0), &p2(t1), &p2(t2))
}

/// Signed area (2x) of triangle `a,b,c` — the 2D cross product `ab x ac`.
#[inline]
fn orient2d(a: &Vector2f, b: &Vector2f, c: &Vector2f) -> Float {
    (b - a).perp(&(c - a))
}

#[inline]
fn point_in_tri_2d(p: &Vector2f, a: &Vector2f, b: &Vector2f, c: &Vector2f) -> bool {
    let d1 = orient2d(p, a, b);
    let d2 = orient2d(p, b, c);
    let d3 = orient2d(p, c, a);
    let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(neg && pos)
}

#[inline]
fn segs_cross_2d(p1: &Vector2f, p2: &Vector2f, p3: &Vector2f, p4: &Vector2f) -> bool {
    let d1 = orient2d(p3, p4, p1);
    let d2 = orient2d(p3, p4, p2);
    let d3 = orient2d(p1, p2, p3);
    let d4 = orient2d(p1, p2, p4);
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

fn seg_tri_overlap_2d(a: &Vector2f, b: &Vector2f, c0: &Vector2f, c1: &Vector2f, c2: &Vector2f) -> bool {
    if point_in_tri_2d(a, c0, c1, c2) || point_in_tri_2d(b, c0, c1, c2) {
        return true;
    }
    segs_cross_2d(a, b, c0, c1) || segs_cross_2d(a, b, c1, c2) || segs_cross_2d(a, b, c2, c0)
}

/// Closest point to `p` on the segment `[a,b]`.
#[inline]
pub fn closest_point_on_segment(p: &Vector3f, a: &Vector3f, b: &Vector3f) -> Vector3f {
    let ab = b - a;
    let denom = ab.dot(&ab);
    if denom < EPS {
        return *a;
    }
    let t = ((p - a).dot(&ab) / denom).clamp(0.0, 1.0);
    a + ab * t
}

/// Axis-aligned bounding box of three points.
#[inline]
pub fn tri_aabb(a: &Vector3f, b: &Vector3f, c: &Vector3f) -> (Vector3f, Vector3f) {
    let lo = a.inf(b).inf(c);
    let hi = a.sup(b).sup(c);
    (lo, hi)
}

/// Axis-aligned bounding box of a segment `[s0,s1]`.
#[inline]
pub fn seg_aabb_pair(s0: &Vector3f, s1: &Vector3f) -> (Vector3f, Vector3f) {
    (s0.inf(s1), s0.sup(s1))
}

/// Do two AABBs (given as min/max corners) overlap, inflated by `margin`?
#[inline]
pub fn aabb_overlap(lo1: &Vector3f, hi1: &Vector3f, lo2: &Vector3f, hi2: &Vector3f, margin: Float) -> bool {
    for i in 0..3 {
        if hi1[i] + margin < lo2[i] || hi2[i] + margin < lo1[i] {
            return false;
        }
    }
    true
}
