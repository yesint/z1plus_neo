//! Building the internal chain representation from either a `molar` system
//! (structure + trajectory) or a Z1-formatted benchmark file.
//!
//! Internal representation: one `Vec<Vector3f>` per chain, holding the
//! *unfolded* node coordinates (no periodic jumps between consecutive beads),
//! plus the periodic box for the frame.

use anyhow::{bail, Context, Result};
use molar::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::Path;

/// How selected backbone beads are partitioned into chains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grouping {
    /// Pick automatically: `Bonds` if the topology has bonds (e.g. from a
    /// `.tpr`), otherwise `Residue` (e.g. a bare `.gro`).
    Auto,
    /// One chain per residue (works for CG melts where residue == molecule).
    Residue,
    /// One chain per connected component of the bond graph, ordered along the
    /// chain. The correct choice whenever real bonds are available.
    Bonds,
    /// One chain per `Topology.molecules` entry. NOTE: for a GROMACS `.tpr`
    /// these are molecule *blocks* (a block with `nmol>1` spans several
    /// chains), so this merges chains — prefer `Bonds`.
    Molecule,
}

impl std::str::FromStr for Grouping {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Grouping::Auto),
            "residue" | "resid" | "res" => Ok(Grouping::Residue),
            "bonds" | "bond" => Ok(Grouping::Bonds),
            "molecule" | "mol" => Ok(Grouping::Molecule),
            other => {
                bail!("unknown grouping '{other}', expected auto|residue|bonds|molecule")
            }
        }
    }
}

/// Ordered global bead indices for every chain. Computed once (topology is
/// constant across a trajectory); positions are read per frame.
pub fn build_chain_indices(sys: &System, sel_str: &str, group: Grouping) -> Result<Vec<Vec<usize>>> {
    let sel = sys
        .select(sel_str)
        .with_context(|| format!("parsing backbone selection '{sel_str}'"))?;
    // Splitting/attribute queries require a state-bound selection.
    let bound = sys.bind(&sel);

    // Resolve Auto based on available connectivity.
    let group = match group {
        Grouping::Auto => {
            if sys.topology().bonds.is_empty() {
                Grouping::Residue
            } else {
                Grouping::Bonds
            }
        }
        g => g,
    };

    let chains: Vec<Vec<usize>> = match group {
        Grouping::Auto => unreachable!("resolved above"),
        Grouping::Residue => bound
            .split_resindex()
            .map(|s| s.iter_index().collect::<Vec<usize>>())
            .collect(),
        Grouping::Bonds => {
            let sel_idx: Vec<usize> = bound.iter_index().collect();
            chains_from_bonds(&sel_idx, &sys.topology().bonds)?
        }
        Grouping::Molecule => {
            let sel_idx: Vec<usize> = bound.iter_index().collect();
            let mols = &sys.topology().molecules;
            if mols.is_empty() {
                bail!("grouping=molecule requested but topology has no molecules \
                       (a .gro file carries no connectivity; use a .tpr, or grouping=residue)");
            }
            let mut out = Vec::with_capacity(mols.len());
            for m in mols {
                let (lo, hi) = (m[0], m[1]);
                let members: Vec<usize> =
                    sel_idx.iter().copied().filter(|&i| i >= lo && i <= hi).collect();
                if !members.is_empty() {
                    out.push(members);
                }
            }
            out
        }
    };

    if chains.is_empty() {
        bail!("selection '{sel_str}' produced no chains");
    }
    Ok(chains)
}

/// Build chains as connected components of the bond graph (restricted to the
/// selected beads), each ordered by walking the linear backbone from an
/// endpoint. This is robust to how a `.tpr` groups molecules into blocks.
fn chains_from_bonds(sel_idx: &[usize], bonds: &[Bond]) -> Result<Vec<Vec<usize>>> {
    let sel_set: HashSet<usize> = sel_idx.iter().copied().collect();

    let mut adj: HashMap<usize, Vec<usize>> = HashMap::with_capacity(sel_set.len());
    for &a in sel_idx {
        adj.entry(a).or_default();
    }
    for b in bonds {
        let [i, j] = b.pair();
        if sel_set.contains(&i) && sel_set.contains(&j) {
            adj.get_mut(&i).unwrap().push(j);
            adj.get_mut(&j).unwrap().push(i);
        }
    }

    let mut atoms: Vec<usize> = sel_idx.to_vec();
    atoms.sort_unstable();
    atoms.dedup();

    let mut visited: HashSet<usize> = HashSet::with_capacity(atoms.len());
    let mut chains: Vec<Vec<usize>> = Vec::new();

    for &start in &atoms {
        if visited.contains(&start) {
            continue;
        }
        // Collect the connected component (DFS).
        let mut comp: Vec<usize> = Vec::new();
        let mut stack = vec![start];
        visited.insert(start);
        while let Some(u) = stack.pop() {
            comp.push(u);
            for &v in &adj[&u] {
                if visited.insert(v) {
                    stack.push(v);
                }
            }
        }
        chains.push(order_component(&comp, &adj));
    }

    if chains.is_empty() {
        bail!("no chains found from bond graph");
    }
    Ok(chains)
}

/// Order a connected component along its linear backbone: pick a degree-1
/// endpoint and walk to the far end. Falls back to sorted-index order for
/// non-path components (branches or rings), with a warning.
fn order_component(comp: &[usize], adj: &HashMap<usize, Vec<usize>>) -> Vec<usize> {
    let sorted = || {
        let mut s = comp.to_vec();
        s.sort_unstable();
        s
    };

    let has_branch = comp.iter().any(|a| adj[a].len() > 2);
    let mut endpoints: Vec<usize> = comp.iter().copied().filter(|a| adj[a].len() == 1).collect();
    endpoints.sort_unstable();

    if has_branch || endpoints.is_empty() {
        log::warn!(
            "chain component of {} beads is not a simple path (branch/ring); using index order",
            comp.len()
        );
        return sorted();
    }

    let mut ordered = Vec::with_capacity(comp.len());
    let mut prev: Option<usize> = None;
    let mut cur = endpoints[0];
    loop {
        ordered.push(cur);
        match adj[&cur].iter().copied().find(|&n| Some(n) != prev) {
            Some(n) if !ordered.contains(&n) => {
                prev = Some(cur);
                cur = n;
            }
            _ => break,
        }
    }

    if ordered.len() == comp.len() {
        ordered
    } else {
        sorted()
    }
}

/// Read the current frame's positions for each chain and unfold them across
/// periodic boundaries so consecutive beads never jump by a box vector.
pub fn frame_from_state(state: &State, chains_idx: &[Vec<usize>]) -> Vec<Vec<Vector3f>> {
    let pbox = state.get_box();
    chains_idx
        .iter()
        .map(|idx| unfold_indices(state, idx, pbox))
        .collect()
}

/// Write one frame's chains as a Z1-format snapshot (readable by both `read_z1`
/// and the Fortran Z1+ binary), so identical input can be fed to both:
///   line 1: number of chains
///   line 2: box extents Lx Ly Lz
///   line 3: C chain lengths
///   then all bead coordinates, chain by chain (x y z per line).
/// Coordinates are written verbatim (unfolded); Z1+ folds internally.
pub fn write_z1(
    path: impl AsRef<Path>,
    chains: &[Vec<Vector3f>],
    box_extents: [Float; 3],
) -> Result<()> {
    use std::io::Write;
    let f = std::fs::File::create(path.as_ref())
        .with_context(|| format!("create {}", path.as_ref().display()))?;
    let mut w = std::io::BufWriter::new(f);
    writeln!(w, "{}", chains.len())?;
    writeln!(w, "{} {} {}", box_extents[0], box_extents[1], box_extents[2])?;
    let mut lens = String::new();
    for c in chains {
        lens.push_str(&c.len().to_string());
        lens.push(' ');
    }
    writeln!(w, "{}", lens.trim_end())?;
    for c in chains {
        for p in c {
            writeln!(w, "{:.6} {:.6} {:.6}", p.x, p.y, p.z)?;
        }
    }
    Ok(())
}

fn unfold_indices(state: &State, idx: &[usize], pbox: Option<&PeriodicBox>) -> Vec<Vector3f> {
    let mut nodes: Vec<Vector3f> = Vec::with_capacity(idx.len());
    for (k, &i) in idx.iter().enumerate() {
        let p = state.get_pos(i).expect("bead index out of range");
        if k == 0 {
            nodes.push(p.coords);
        } else {
            let prev: Pos = nodes[k - 1].into();
            let unf = match pbox {
                Some(b) => b.closest_image(p, &prev).coords,
                None => p.coords,
            };
            nodes.push(unf);
        }
    }
    nodes
}

/// A configuration loaded from a Z1-formatted file (benchmark inputs).
pub struct Z1Config {
    pub chains: Vec<Vec<Vector3f>>,
    pub pbox: PeriodicBox,
}

/// Parse a Z1-formatted configuration file.
///
/// Layout: line 1 = number of chains `C`; line 2 = box `bx by bz`
/// (orthorhombic); line 3 = `C` chain lengths `N_1 .. N_C`; then
/// `sum(N_j)` lines of `x y z`. Coordinates are unfolded defensively.
pub fn read_z1(path: impl AsRef<Path>) -> Result<Z1Config> {
    let path = path.as_ref();
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = std::io::BufReader::new(f);

    let mut tokens = TokenStream::new(&mut reader);

    let n_chains: usize = tokens.next_parse().context("reading number of chains")?;
    let bx: Float = tokens.next_parse().context("reading box x")?;
    let by: Float = tokens.next_parse().context("reading box y")?;
    let bz: Float = tokens.next_parse().context("reading box z")?;

    let mut lengths = Vec::with_capacity(n_chains);
    for _ in 0..n_chains {
        lengths.push(tokens.next_parse::<usize>().context("reading chain length")?);
    }

    let mut mat = Matrix3f::zeros();
    mat[(0, 0)] = bx;
    mat[(1, 1)] = by;
    mat[(2, 2)] = bz;
    let pbox = PeriodicBox::from_matrix(mat).context("constructing periodic box")?;

    let mut chains = Vec::with_capacity(n_chains);
    for &n in &lengths {
        let mut nodes = Vec::with_capacity(n);
        for k in 0..n {
            let x: Float = tokens.next_parse()?;
            let y: Float = tokens.next_parse()?;
            let z: Float = tokens.next_parse()?;
            let raw = Vector3f::new(x, y, z);
            if k == 0 {
                nodes.push(raw);
            } else {
                let prev: Pos = nodes[k - 1].into();
                nodes.push(pbox.closest_image(&raw.into(), &prev).coords);
            }
        }
        chains.push(nodes);
    }

    Ok(Z1Config { chains, pbox })
}

/// Minimal whitespace token stream over a buffered reader (Z1 files freely mix
/// line breaks and spaces).
struct TokenStream<'a, R: BufRead> {
    reader: &'a mut R,
    buf: Vec<String>,
    pos: usize,
}

impl<'a, R: BufRead> TokenStream<'a, R> {
    fn new(reader: &'a mut R) -> Self {
        Self { reader, buf: Vec::new(), pos: 0 }
    }

    fn next_token(&mut self) -> Result<String> {
        loop {
            if self.pos < self.buf.len() {
                let t = std::mem::take(&mut self.buf[self.pos]);
                self.pos += 1;
                return Ok(t);
            }
            let mut line = String::new();
            let n = self.reader.read_line(&mut line)?;
            if n == 0 {
                bail!("unexpected end of Z1 file");
            }
            self.buf = line.split_whitespace().map(|s| s.to_owned()).collect();
            self.pos = 0;
        }
    }

    fn next_parse<T: std::str::FromStr>(&mut self) -> Result<T>
    where
        T::Err: std::fmt::Display,
    {
        let t = self.next_token()?;
        t.parse::<T>().map_err(|e| anyhow::anyhow!("parsing '{t}': {e}"))
    }
}
