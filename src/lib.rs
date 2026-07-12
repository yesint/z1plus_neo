//! `entangl_rs` — geometric topological analysis of polymer entanglements.
//!
//! Reimplementation of the Z1 / Z1+ "shortest multiple disconnected path" (SMDP)
//! idea (Kröger, Comput. Phys. Commun. 168 (2005) 209; 283 (2023) 108567):
//! fix all chain ends, forbid chain crossings, and monotonically shrink every
//! chain's contour until it becomes a set of straight segments joined at kinks.
//! Remaining interior kinks are the entanglement points; their count per chain
//! is `Z`, and the summed segment length is the primitive-path length `Lpp`.
//!
//! Molecular I/O, selections, PBC and geometry come from `molar`.

pub mod cells;
pub mod chain;
pub mod geom;
pub mod grid;
pub mod linking;
pub mod pool;
pub mod report;
pub mod sweep;
pub mod z1;
pub mod z1geom;
pub mod z1plus;
