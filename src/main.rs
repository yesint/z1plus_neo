//! `entangl_rs` — CLI for entanglement analysis of a structure + trajectory
//! using any file formats `molar` accepts (gro/xtc/trr/pdb/tpr/...).
//!
//! Uses the Z1+ SMDP core (`sweep::Network`), validated against the
//! Z1+ binary (benchmarks 01-04 exact; melts within a few % on <Z>, <Lpp> within
//! ~2%). Usage:
//! `entangl_rs -f struct.tpr traj.xtc [--sel <expr>] [--group bonds] [--export-z1 dir]`

use anyhow::{anyhow, Result};
use clap::Args;
use molar::prelude::*;

use entangl_rs::chain::{build_chain_indices, frame_from_state, write_z1, Grouping};
use entangl_rs::report::Reporter;
use entangl_rs::cells::BoxDims;
use entangl_rs::sweep::analyze_chains;
use entangl_rs::z1::{ChainResult, FrameResult};

#[derive(Args, Debug, Clone)]
struct Flags {
    /// Selection defining the polymer backbone beads (one bead == one node)
    #[arg(short = 's', long = "sel", default_value = "all")]
    sel: String,

    /// How to partition selected beads into chains: auto | residue | bonds | molecule
    #[arg(short = 'g', long = "group", default_value = "auto")]
    group: Grouping,

    /// Output file prefix
    #[arg(short = 'o', long = "out", default_value = "entangl")]
    out: String,

    /// Chain thickness (Z1+ `thickness`; contacts rest this far off obstacles)
    #[arg(long = "thickness", default_value_t = 0.002)]
    thickness: f64,

    /// lmax factor (Z1+ `lmax_factor`; lmax = factor * max initial bond)
    #[arg(long = "lmax-factor", default_value_t = 1.0)]
    lmax_factor: f64,

    /// Also detect self-entanglements (a chain crossing itself)
    #[arg(long = "self", default_value_t = false)]
    self_entanglement: bool,

    /// Safety cap on minimization sweeps
    #[arg(long = "max-sweeps", default_value_t = 5000)]
    max_sweeps: usize,

    /// Export each analyzed frame's chains as a Z1-format file into this dir
    /// (so the same input can be fed to the Fortran Z1+ binary for comparison).
    #[arg(long = "export-z1")]
    export_z1: Option<String>,
}

struct EntanglTask {
    chains_idx: Vec<Vec<usize>>,
    thickness: f64,
    lmax_factor: f64,
    max_sweeps: usize,
    export_z1: Option<String>,
    frame_no: usize,
    reporter: Reporter,
    out_prefix: String,
}

impl AnalysisTask<Flags> for EntanglTask {
    fn task_name() -> String {
        "entangl_rs".to_owned()
    }

    fn new(context: &mut AnalysisContext<Flags>) -> Result<Self> {
        let top = context.sys.topology();
        log::info!(
            "system: {} atoms, {} bonds, {} molecules",
            context.sys.select_all().len(),
            top.bonds.len(),
            top.molecules.len()
        );

        let chains_idx = build_chain_indices(&context.sys, &context.args.sel, context.args.group)?;
        let n_true = chains_idx.iter().filter(|c| c.len() >= 3).count();
        let (min_len, max_len) = chains_idx
            .iter()
            .map(|c| c.len())
            .fold((usize::MAX, 0), |(mn, mx), l| (mn.min(l), mx.max(l)));
        log::info!(
            "backbone '{}' grouped by {:?}: {} chains ({} true), length {}..{} beads",
            context.args.sel,
            context.args.group,
            chains_idx.len(),
            n_true,
            min_len,
            max_len
        );
        if context.args.self_entanglement {
            log::warn!("--self is not yet supported by the current core; ignoring");
        }
        if let Some(dir) = &context.args.export_z1 {
            std::fs::create_dir_all(dir)?;
            log::info!("exporting each frame as Z1-format into {dir}/");
        }

        Ok(Self {
            chains_idx,
            thickness: context.args.thickness,
            lmax_factor: context.args.lmax_factor,
            max_sweeps: context.args.max_sweeps,
            export_z1: context.args.export_z1.clone(),
            frame_no: 0,
            reporter: Reporter::new(&context.args.out)?,
            out_prefix: context.args.out.clone(),
        })
    }

    fn process_frame(&mut self, context: &mut AnalysisContext<Flags>) -> Result<()> {
        let state = context.sys.state();
        let chains_v = frame_from_state(state, &self.chains_idx); // unfolded per chain
        let pbox = state
            .get_box()
            .ok_or_else(|| anyhow!("frame has no periodic box; SMDP needs one"))?;
        let ext = pbox.get_box_extents();
        let bx = BoxDims {
            l: [ext.x as f64, ext.y as f64, ext.z as f64],
        };

        if let Some(dir) = &self.export_z1 {
            let path = format!("{dir}/frame_{:05}.Z1", self.frame_no);
            write_z1(&path, &chains_v, [ext.x, ext.y, ext.z])?;
        }

        // f64 chains for the SMDP core
        let chains: Vec<Vec<[f64; 3]>> = chains_v
            .iter()
            .map(|c| c.iter().map(|p| [p.x as f64, p.y as f64, p.z as f64]).collect())
            .collect();
        let stats = analyze_chains(&chains, bx, self.thickness, self.lmax_factor, self.max_sweeps);
        let fr = FrameResult {
            chains: stats
                .into_iter()
                .map(|(n, is_true, z, lpp, ree)| ChainResult {
                    n_beads: n,
                    is_true,
                    z: z as usize,
                    lpp: lpp as Float,
                    ree: ree as Float,
                })
                .collect(),
        };
        self.reporter.add_frame(&fr)?;
        self.frame_no += 1;
        Ok(())
    }

    fn post_process(&mut self, _context: &mut AnalysisContext<Flags>) -> Result<()> {
        let means = self.reporter.finalize()?;
        log::info!("{}", means.report(std::path::Path::new(&self.out_prefix)));
        Ok(())
    }
}

fn main() -> Result<()> {
    env_logger::builder()
        .format_timestamp(None)
        .format_indent(Some(8))
        .filter_level(log::LevelFilter::Info)
        .init();
    EntanglTask::run()?;
    Ok(())
}
