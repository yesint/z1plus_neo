//! `z1run` — run the entanglement analysis directly on Z1-formatted files.
//!
//! This bypasses `molar` I/O and is used to validate `entangl_rs` against the
//! reference Z1+ binary on the published benchmark configurations.
//!
//! Usage: `z1run [--self] file1.Z1 [file2.Z1 ...]`

use anyhow::Result;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use entangl_rs::chain::read_z1;
use entangl_rs::report::z1_estimators;
use entangl_rs::z1::{analyze_frame, Options, MIN_TRUE_CHAIN};

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum Algorithm {
    /// Previous remove/slide implementation.
    Legacy,
    /// Insertion-based implementation of the Z1+ core.
    Z1Plus,
}

#[derive(Parser, Debug)]
#[command(name = "z1run", about = "entangl_rs on Z1-formatted benchmark files")]
struct Cli {
    /// Z1-formatted configuration files
    #[arg(required = true)]
    files: Vec<PathBuf>,

    /// Minimizer implementation
    #[arg(long, value_enum, default_value_t = Algorithm::Legacy)]
    algorithm: Algorithm,

    /// Also detect self-entanglements
    #[arg(long = "self", default_value_t = false)]
    self_entanglement: bool,

    /// Safety cap on minimization sweeps
    #[arg(long = "max-sweeps", default_value_t = 2000)]
    max_sweeps: usize,

    /// Max segment length (ghost-node cap). Default: 0.2 * min box extent.
    #[arg(long = "lmax")]
    lmax: Option<f32>,

    /// Kink angle threshold in degrees
    #[arg(long = "kink-deg", default_value_t = 2.9)]
    kink_deg: f32,

    /// Disable node sliding (removal only; diagnostic)
    #[arg(long = "no-slide", default_value_t = false)]
    no_slide: bool,

    /// Force brute-force neighbor scan (disable spatial grid; diagnostic)
    #[arg(long = "brute", default_value_t = false)]
    brute: bool,

    /// Disable node removal (slide only; diagnostic)
    #[arg(long = "no-remove", default_value_t = false)]
    no_remove: bool,

    /// Report total |Gauss linking| before/after minimization (topology check)
    #[arg(long = "check-linking", default_value_t = false)]
    check_linking: bool,

    /// Print per-chain (index, N, Ree, Lpp, Z)
    #[arg(long = "dump", default_value_t = false)]
    dump: bool,

    /// Chain thickness (min gap kept from other chains)
    #[arg(long = "thickness", default_value_t = 0.01)]
    thickness: f32,

    /// Use chord-foot tightening (Z1-style maximal displacement) vs wrap-point
    #[arg(long = "chord-slide", default_value_t = false)]
    chord_slide: bool,

    /// Use thickness-fat removal instead of strict (diagnostic)
    #[arg(long = "fat-removal", default_value_t = false)]
    fat_removal: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if std::env::var("ENTANGL_VERIFY").is_ok() {
        entangl_rs::z1::VERIFY.with(|v| v.set(true));
    }
    let opts = Options {
        self_entanglement: cli.self_entanglement,
        max_sweeps: cli.max_sweeps,
        kink_angle: cli.kink_deg.to_radians() as molar::prelude::Float,
        lmax: cli
            .lmax
            .map(|x| x as molar::prelude::Float)
            .unwrap_or(molar::prelude::Float::INFINITY),
        slide: !cli.no_slide,
        use_grid: !cli.brute,
        remove: !cli.no_remove,
        thickness: cli.thickness as molar::prelude::Float,
        chord_slide: cli.chord_slide,
        strict_removal: !cli.fat_removal,
    };

    for path in &cli.files {
        let cfg = read_z1(path)?;
        let n_chains = cfg.chains.len();
        let ext = cfg.pbox.get_box_extents();

        if cli.check_linking && cli.algorithm == Algorithm::Legacy {
            let lk_before = entangl_rs::linking::total_abs_linking(&cfg.chains, Some(&cfg.pbox));
            let minimized = entangl_rs::z1::minimize_chains(&cfg.chains, Some(&cfg.pbox), opts);
            let lk_after = entangl_rs::linking::total_abs_linking(&minimized, Some(&cfg.pbox));
            println!(
                "  linking: before={lk_before:.3} after={lk_after:.3}  (drop => chains crossed)"
            );
        }

        let t0 = std::time::Instant::now();
        let fr = match cli.algorithm {
            Algorithm::Legacy => analyze_frame(&cfg.chains, Some(&cfg.pbox), opts),
            Algorithm::Z1Plus => entangl_rs::z1plus::analyze_frame(
                &cfg.chains,
                Some(&cfg.pbox),
                entangl_rs::z1plus::Options {
                    self_entanglement: cli.self_entanglement,
                    max_sweeps: cli.max_sweeps,
                    thickness: cli.thickness as f64,
                    lmax: cli.lmax.map(f64::from).unwrap_or(f64::INFINITY),
                },
            ),
        };
        let elapsed = t0.elapsed();

        // Aggregate over true chains.
        let mut n_true = 0usize;
        let mut violations = 0usize; // chains with Lpp < Ree (impossible => bug)
        let (mut sum_n, mut sum_z, mut sum_lpp, mut sum_lpp2, mut sum_ree2) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for c in &fr.chains {
            if c.n_beads < MIN_TRUE_CHAIN {
                continue;
            }
            n_true += 1;
            sum_n += c.n_beads as f64;
            sum_z += c.z as f64;
            sum_lpp += c.lpp as f64;
            sum_lpp2 += (c.lpp as f64).powi(2);
            sum_ree2 += (c.ree as f64) * (c.ree as f64);
            if c.lpp + 1.0e-3 < c.ree {
                violations += 1;
            }
        }
        if cli.dump {
            for (idx, c) in fr.chains.iter().enumerate() {
                println!(
                    "  chain {:>3}: N={:>4} Ree={:>8.3} Lpp={:>8.3} Z={}",
                    idx + 1,
                    c.n_beads,
                    c.ree,
                    c.lpp,
                    c.z
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
            if violations > 0 { format!("  [!! {violations} Lpp<Ree violations]") } else { String::new() }
        );
    }

    Ok(())
}
