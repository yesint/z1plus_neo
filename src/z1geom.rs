//! Core geometry primitive for the SMDP move.
//!
//! `get_t1_t2_ts` is the per-node obstacle predicate: "does obstacle segment
//! (S1,S2) pierce the swept triangle (A,B,C) as B relaxes toward the chord?" It
//! returns the pierce point P plus the binding metric `cos(AB, A-P)`. All
//! arithmetic is f64; the degeneracy cut is 1e-30 (applied to the three
//! denominators D,E,G) and the parameter-range test is strict `0<=·<=1` with no
//! slack epsilon.

pub type V3 = [f64; 3];

#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
/// Scalar triple product `(u × v) · w`.
#[inline]
pub fn crossdot(u: V3, v: V3, w: V3) -> f64 {
    dot(cross(u, v), w)
}
#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// Degeneracy zero-cut on the triple-product denominators.
pub const DEGEN: f64 = 1.0e-30;

/// Result of the obstacle/triangle pierce test.
#[derive(Clone, Copy, Debug)]
pub struct Pierce {
    /// Pierce point in the (A,B,C) triangle.
    pub p: V3,
    /// Binding metric `cos(AB, A-P)` used to select the obstacle to wrap.
    pub cos: f64,
    /// Obstacle parameter `t2` along S1->S2 (for the contact's material coord).
    pub t2: f64,
}

/// The obstacle/triangle pierce test `get_t1_t2_ts(S1,S2,A,B,C)`.
/// Returns `Some(Pierce)` iff the obstacle segment pierces the triangle
/// (all three parameters in `[0,1]`), else `None`.
pub fn get_t1_t2_ts(s1: V3, s2: V3, a: V3, b: V3, c: V3) -> Option<Pierce> {
    let s21 = sub(s2, s1);
    let q = sub(s1, b);
    let ab = sub(a, b);
    let cb = sub(c, b);
    let ca = sub(c, a); // = cb - ab

    let t_a = crossdot(s21, q, cb); // numerator of ts
    let d = crossdot(ca, s21, q); //   denom of ts
    if d.abs() < DEGEN {
        return None;
    }
    let e = crossdot(cb, s21, ab); // denom of t1
    if e.abs() < DEGEN {
        return None;
    }
    let f = crossdot(cb, q, ab); // numerator of t2
    let g = crossdot(s21, cb, ab); // denom of t2 (= -E)
    if g.abs() < DEGEN {
        return None;
    }

    let ts = t_a / d;
    let t1 = d / e;
    let t2 = f / g;
    // strict range test, no epsilon
    let in01 = |x: f64| (0.0..=1.0).contains(&x);
    if !(in01(ts) && in01(t1) && in01(t2)) {
        return None;
    }

    // pierce point P = B + t1*( ts*AB + (1-ts)*CB )
    let dir = add(scale(ab, ts), scale(cb, 1.0 - ts));
    let p = add(b, scale(dir, t1));

    // binding metric cos(AB, A-P)
    let ap = sub(a, p);
    let denom = norm(ab) * norm(ap);
    let cos = if denom > 0.0 {
        (dot(ab, ap) / denom).clamp(-1.0, 1.0)
    } else {
        // degenerate; treat as non-binding (never selected: best requires cos<1)
        -1.0
    };
    Some(Pierce { p, cos, t2 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: V3, b: V3, tol: f64) -> bool {
        sub(a, b).iter().all(|d| d.abs() < tol)
    }

    #[test]
    fn obstacle_through_triangle_interior_crosses() {
        // triangle apex B=(0,0,0), A=(0,1,0), C=(1,0,0) in the z=0 plane
        let b = [0.0, 0.0, 0.0];
        let a = [0.0, 1.0, 0.0];
        let c = [1.0, 0.0, 0.0];
        // obstacle piercing the plane at (0.2,0.2,0), interior of the triangle
        let s1 = [0.2, 0.2, -1.0];
        let s2 = [0.2, 0.2, 1.0];
        let r = get_t1_t2_ts(s1, s2, a, b, c).expect("should cross");
        assert!(close(r.p, [0.2, 0.2, 0.0], 1e-9), "P={:?}", r.p);
        assert!((r.t2 - 0.5).abs() < 1e-9, "t2={}", r.t2); // midpoint of obstacle
                                                           // cos(AB, A-P): AB=A-B=(0,1,0); A-P=(-0.2,0.8,0); cos = 0.8/|A-P|
        let expect_cos = 0.8 / ((0.2f64).powi(2) + (0.8f64).powi(2)).sqrt();
        assert!((r.cos - expect_cos).abs() < 1e-9, "cos={}", r.cos);
    }

    #[test]
    fn obstacle_missing_triangle_does_not_cross() {
        let b = [0.0, 0.0, 0.0];
        let a = [0.0, 1.0, 0.0];
        let c = [1.0, 0.0, 0.0];
        // pierces the plane at (0.8,0.8,0): x+y=1.6 > 1 → outside the triangle
        let s1 = [0.8, 0.8, -1.0];
        let s2 = [0.8, 0.8, 1.0];
        assert!(get_t1_t2_ts(s1, s2, a, b, c).is_none());
    }

    #[test]
    fn obstacle_not_reaching_plane_does_not_cross() {
        let b = [0.0, 0.0, 0.0];
        let a = [0.0, 1.0, 0.0];
        let c = [1.0, 0.0, 0.0];
        // both endpoints on the same side of z=0 → obstacle param outside [0,1]
        let s1 = [0.2, 0.2, 0.5];
        let s2 = [0.2, 0.2, 1.5];
        assert!(get_t1_t2_ts(s1, s2, a, b, c).is_none());
    }

    #[test]
    fn obstacle_parallel_to_plane_is_degenerate() {
        let b = [0.0, 0.0, 0.0];
        let a = [0.0, 1.0, 0.0];
        let c = [1.0, 0.0, 0.0];
        // obstacle lies in the z=0.5 plane, parallel to the triangle
        let s1 = [-1.0, 0.2, 0.5];
        let s2 = [1.0, 0.2, 0.5];
        assert!(get_t1_t2_ts(s1, s2, a, b, c).is_none());
    }

    #[test]
    fn crossdot_is_scalar_triple_product() {
        let u = [1.0, 0.0, 0.0];
        let v = [0.0, 1.0, 0.0];
        let w = [0.0, 0.0, 1.0];
        assert!((crossdot(u, v, w) - 1.0).abs() < 1e-15);
        assert!((crossdot(v, u, w) + 1.0).abs() < 1e-15); // swap → sign flip
    }
}
