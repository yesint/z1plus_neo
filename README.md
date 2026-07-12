# z1plus_neo

An open Rust implementation of the **Z1+ "shortest multiple disconnected path"
(SMDP)** method for topological analysis of polymer entanglements, validated to
match the official Z1+ reference code.

Z1+ (M. Kröger et al., *Comput. Phys. Commun.* **283** (2023) 108567; successor
of Z1, *ibid.* **168** (2005) 209) is a standard geometric tool for counting
entanglements in polymer melts, distributed as a compiled binary. `z1plus_neo`
is an independent, owned, and modifiable implementation of the same geometry,
built on the [`molar`](https://github.com/yesint/molar) molecular library for
IO / selections / periodic boundaries / geometry, with `nalgebra` for math.

## What it computes

Given a polymer configuration (linear chains of beads in a periodic box), the
SMDP method fixes each chain's two endpoints, forbids chains from crossing one
another, and monotonically shrinks every chain's contour to its shortest length.
The result is a set of straight segments meeting at kinks; each kink is an
entanglement. Per frame it reports:

- **Z** — mean number of entanglements (kinks) per chain
- **Lpp** — mean primitive-path contour length
- **Ree** — mean end-to-end distance (an input property; algorithm-independent)
- **Ne** estimators (Hoy et al. 2009): classical/modified kink & coil.

## Validation

Results are checked directly against the official Z1+ binary (used as the
reference oracle), feeding both codes identical input.

**Distribution benchmarks** — error vs the Z1+ reference (`d<Z>` / `d<Lpp>`):

| benchmark | system                    | d`<Z>`  | d`<Lpp>` |
|-----------|---------------------------|---------|----------|
| 01–04     | single chain + obstacles  | exact   | exact    |
| 05        | melt (50×20)              | 0.00 %  | +0.02 %  |
| 07        | melt (129×100)            | +0.52 % | −0.02 %  |
| 08        | melt (1032×400)           | −0.48 % | +0.05 %  |
| 10        | melt (100×200)            | −0.80 % | +0.86 %  |
| 12        | melt (37×819)             | −0.78 % | +0.29 %  |
| 13        | melt (128×1024)           | −0.52 % | −0.04 %  |
| 14        | dense melt (155×183)      | −2.3 %  | +0.83 %  |

Benchmarks 01–04 are exact on both `Z` and `Lpp` and are enforced by a
regression test (`simple_benchmarks_stay_exact`). Across the melts both
quantities agree to within ~1 %, with one exception: bench-14's `<Z>` sits ~2.3 %
low because in very dense tangles many equally-short primitive paths exist and
the two codes select different (equally valid) ones; matching Z1+'s exact choice
would require its non-public path-ordering. (Bench-11 is a near-unentangled
degenerate case, oracle `<Z>`≈0.05, and is excluded.)

**Real system** — a polydisperse polyethylene melt (85 chains, 5–235 beads,
cubic box ~19 nm), same input to both codes over 11 frames (0–100 ns):

| quantity | z1plus_neo | Z1+   | agreement |
|----------|-----------:|------:|-----------|
| `<Z>`    | 4.175      | 4.171 | **0.10 %** |
| `<Lpp>`  | 10.900     | 10.873| **0.25 %** |

## Build

```sh
cargo build --release --offline --locked
```

Reading a GROMACS `.tpr` topology (for bond-based chain grouping) needs `molar`'s
gromacs plugin, which requires `.cargo/config.toml` with `GROMACS_SOURCE_DIR` /
`GROMACS_BUILD_DIR` / `GROMACS_LIB_DIR` — copy and edit
[`.cargo/config.toml.example`](.cargo/config.toml.example). Without them, `.gro`
+ `.xtc` still work (residue-based grouping only).

## Usage

```sh
# any structure + trajectory molar reads: gro / xtc / trr / pdb / tpr
entangl_rs -f system.tpr traj.xtc -g bonds -o out
```

Key options:

- `-g auto|residue|bonds|molecule` — how to partition beads into chains. For a
  melt with a `.tpr`, use `bonds` (connected components of the bond graph);
  a bare `.gro` (no bonds) must use `residue`.
- `--thickness <t>` (default 0.002) and `--lmax-factor <f>` (default 1.0) —
  Z1+'s parameters.
- `--export-z1 <dir>` — dump each analyzed frame as a Z1-format snapshot, so the
  identical input can also be fed to the Z1+ binary for a direct comparison.
- `-b / -e / --skip` — frame range.

Writes `<out>_summary.dat`, `<out>_Z_values.dat`, `<out>_Lpp_values.dat`
(one line per frame, one column per chain).

## Repository layout

```
src/
  main.rs      CLI (molar AnalysisTask) driving the SMDP core
  sweep.rs     the SMDP minimizer: build, minimize, per-node wrap logic, finalize
  pool.rs      flat stable-id node pool (struct-of-arrays + free list)
  cells.rs     Allen–Tildesley linked-cell neighbour list
  z1geom.rs    the core segment/triangle pierce predicate
  chain.rs     chain extraction from a molar System; Z1-format read/write
  report.rs    per-frame accumulation, summary, Ne estimators
tests/fixtures/  Z1+ benchmark configs (01–05, 07, 10, 14) used by the tests
```

Tests: `cargo test --release --offline --locked` (fast; includes the exact 01–04
gate). The melt-benchmark parity check is opt-in:
`cargo test --release --offline --locked melt_benchmarks_parity -- --ignored --nocapture`.

## Attribution & license

This project implements the algorithm of **Z1+** by Martin Kröger, Joseph D.
Dietz, Robert S. Hoy and Clarisse Luap (ETH Zürich; Z1+ is Apache-2.0). If you
use entanglement results from this or Z1+, please cite their work
([doi:10.1016/j.cpc.2022.108567](https://doi.org/10.1016/j.cpc.2022.108567) and
references therein). The benchmark configurations under `tests/fixtures/` are
from the Apache-2.0 Z1+ distribution.

`z1plus_neo` is released under the **Artistic-2.0** license (matching its `molar`
dependency; see the `license` field in `Cargo.toml`).
