//! `z1run` — run the SMDP entanglement analysis directly on Z1-formatted files.
//!
//! Bypasses `molar` trajectory I/O and runs the *same* `sweep` core as the main
//! `entangl_rs` binary, so the published benchmark configurations can be checked
//! against the reference Z1+ binary on identical input.
//!
//! Usage: `z1run [--thickness T] [--lmax-factor F] file1.Z1 [file2.Z1 ...]`

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use entangl_rs::cells::BoxDims;
use entangl_rs::chain::read_z1;
use entangl_rs::report::z1_estimators;
use entangl_rs::sweep::{analyze_chains, MIN_TRUE_CHAIN};

#[derive(Parser, Debug)]
#[command(name = "z1run", about = "entangl_rs SMDP core on Z1-formatted benchmark files")]
struct Cli {
    /// Z1-formatted configuration files
    #[arg(required = true)]
    files: Vec<PathBuf>,

    /// Chain thickness (Z1+ `thickness`; contacts rest this far off obstacles)
    #[arg(long = "thickness", default_value_t = 0.002)]
    thickness: f64,

    /// lmax factor (Z1+ `lmax_factor`; lmax = factor * max initial bond)
    #[arg(long = "lmax-factor", default_value_t = 1.0)]
    lmax_factor: f64,

    /// Safety cap on minimization sweeps
    #[arg(long = "max-sweeps", default_value_t = 5000)]
    max_sweeps: usize,

    /// Print per-chain (index, N, Ree, Lpp, Z)
    #[arg(long = "dump", default_value_t = false)]
    dump: bool,

    /// Also detect self-entanglements (not yet supported by the core)
    #[arg(long = "self", default_value_t = false)]
    self_entanglement: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.self_entanglement {
        eprintln!("warning: --self is not supported by the current core; ignoring");
    }

    for path in &cli.files {
        let cfg = read_z1(path)?;
        let n_chains = cfg.chains.len();
        let ext = cfg.pbox.get_box_extents();
        let bx = BoxDims {
            l: [ext.x as f64, ext.y as f64, ext.z as f64],
        };
        // The sweep core works in f64 `[x,y,z]` coordinates.
        let chains: Vec<Vec<[f64; 3]>> = cfg
            .chains
            .iter()
            .map(|c| c.iter().map(|p| [p.x as f64, p.y as f64, p.z as f64]).collect())
            .collect();

        let t0 = std::time::Instant::now();
        let stats = analyze_chains(&chains, bx, cli.thickness, cli.lmax_factor, cli.max_sweeps);
        let elapsed = t0.elapsed();

        // Aggregate over true chains. `stats` items are
        // (n_beads, is_true, z, lpp, ree).
        let mut n_true = 0usize;
        let mut violations = 0usize; // chains with Lpp < Ree (impossible => bug)
        let (mut sum_n, mut sum_z, mut sum_lpp, mut sum_lpp2, mut sum_ree2) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for &(n_beads, _is_true, z, lpp, ree) in &stats {
            if n_beads < MIN_TRUE_CHAIN {
                continue;
            }
            n_true += 1;
            sum_n += n_beads as f64;
            sum_z += z as f64;
            sum_lpp += lpp;
            sum_lpp2 += lpp * lpp;
            sum_ree2 += ree * ree;
            if lpp + 1.0e-3 < ree {
                violations += 1;
            }
        }
        if cli.dump {
            for (idx, &(n_beads, _is_true, z, lpp, ree)) in stats.iter().enumerate() {
                println!(
                    "  chain {:>3}: N={:>4} Ree={:>8.3} Lpp={:>8.3} Z={}",
                    idx + 1,
                    n_beads,
                    ree,
                    lpp,
                    z
                );
            }
        }

        let s = n_true.max(1) as f64;
        let mean_n = sum_n / s;
        let mean_z = sum_z / s;
        let mean_lpp = sum_lpp / s;
        let mean_lpp2 = sum_lpp2 / s;
        let mean_ree2 = sum_ree2 / s;
        let mean_ree = mean_ree2.sqrt();
        let estimators = z1_estimators(mean_n, mean_z, mean_ree2, mean_lpp, mean_lpp2);

        println!("=== {} ===", path.display());
        println!(
            "  chains={n_chains} true={n_true}  box=[{:.4}, {:.4}, {:.4}]",
            ext.x, ext.y, ext.z
        );
        println!(
            "  <N>={mean_n:.4}  <Ree>={mean_ree:.4}  <Lpp>={mean_lpp:.4}  <Z>={mean_z:.4}  Ne_CK={:.4}  Ne_MK={:.4}  Ne_CC={:.4}  Ne_MC={:.4}  ({:.2?}){}",
            estimators.ne_ck,
            estimators.ne_mk,
            estimators.ne_cc,
            estimators.ne_mc,
            elapsed,
            if violations > 0 {
                format!("  [!! {violations} Lpp<Ree violations]")
            } else {
                String::new()
            }
        );
    }

    Ok(())
}
