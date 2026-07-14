//! Accumulation of per-frame results and writing of summary / per-chain output.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use molar::prelude::Float;

/// Per-chain result for one analyzed frame.
#[derive(Clone, Debug)]
pub struct ChainResult {
    pub n_beads: usize,
    pub is_true: bool,
    pub z: usize,
    pub lpp: Float,
    pub ree: Float,
}

/// Result for one analyzed frame (all chains).
#[derive(Clone, Debug)]
pub struct FrameResult {
    pub chains: Vec<ChainResult>,
}

/// The four single-configuration entanglement-length estimators reported by
/// Z1+ (Hoy, Foteinopoulou & Kröger, Phys. Rev. E 80, 031803).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimators {
    pub ne_ck: f64,
    pub ne_mk: f64,
    pub ne_cc: f64,
    pub ne_mc: f64,
}

/// Evaluate Z1+'s S-estimators from true-chain ensemble means.
///
/// `mean_ree2` and `mean_lpp2` are means of the squared per-chain quantities,
/// not squares of their means. Infinite values for an unentangled ensemble are
/// intentional and match the limiting definitions.
pub fn z1_estimators(
    mean_n: f64,
    mean_z: f64,
    mean_ree2: f64,
    mean_lpp: f64,
    mean_lpp2: f64,
) -> Estimators {
    Estimators {
        ne_ck: mean_n * (mean_n - 1.0) / (mean_n + (mean_n - 1.0) * mean_z),
        ne_mk: mean_n / mean_z,
        ne_cc: (mean_n - 1.0) * mean_ree2 / mean_lpp.powi(2),
        ne_mc: (mean_n - 1.0) * mean_ree2 / (mean_lpp2 - mean_ree2),
    }
}

/// Collects statistics over true chains across all frames and streams per-chain
/// `Z` and `Lpp` values to disk.
pub struct Reporter {
    z_file: BufWriter<File>,
    lpp_file: BufWriter<File>,
    prefix: String,

    frames: usize,
    // Sums over individual true-chain samples (chain x frame).
    samples: f64,
    sum_n: f64,
    sum_ree2: f64,
    sum_lpp: f64,
    sum_lpp2: f64,
    sum_z: f64,
    // Per-frame mean of true chains per frame.
    sum_ntrue: f64,
}

impl Reporter {
    pub fn new(prefix: &str) -> Result<Self> {
        let z_path = format!("{prefix}_Z_values.dat");
        let lpp_path = format!("{prefix}_Lpp_values.dat");
        Ok(Self {
            z_file: BufWriter::new(
                File::create(&z_path).with_context(|| format!("creating {z_path}"))?,
            ),
            lpp_file: BufWriter::new(
                File::create(&lpp_path).with_context(|| format!("creating {lpp_path}"))?,
            ),
            prefix: prefix.to_owned(),
            frames: 0,
            samples: 0.0,
            sum_n: 0.0,
            sum_ree2: 0.0,
            sum_lpp: 0.0,
            sum_lpp2: 0.0,
            sum_z: 0.0,
            sum_ntrue: 0.0,
        })
    }

    /// Record one frame: append per-chain `Z`/`Lpp` lines and update sums.
    pub fn add_frame(&mut self, fr: &FrameResult) -> Result<()> {
        let mut n_true = 0usize;
        for c in &fr.chains {
            if !c.is_true {
                continue;
            }
            n_true += 1;
            self.samples += 1.0;
            self.sum_n += c.n_beads as f64;
            self.sum_ree2 += (c.ree as f64) * (c.ree as f64);
            self.sum_lpp += c.lpp as f64;
            self.sum_lpp2 += (c.lpp as f64) * (c.lpp as f64);
            self.sum_z += c.z as f64;

            write!(self.z_file, "{} ", c.z)?;
            write!(self.lpp_file, "{:.5} ", c.lpp)?;
        }
        writeln!(self.z_file)?;
        writeln!(self.lpp_file)?;
        self.sum_ntrue += n_true as f64;
        self.frames += 1;
        Ok(())
    }

    /// Aggregate means over all true-chain samples.
    pub fn means(&self) -> Means {
        let s = self.samples.max(1.0);
        let mean_n = self.sum_n / s;
        let mean_z = self.sum_z / s;
        let mean_ree2 = self.sum_ree2 / s;
        let mean_lpp = self.sum_lpp / s;
        let mean_lpp2 = self.sum_lpp2 / s;
        let estimators = z1_estimators(mean_n, mean_z, mean_ree2, mean_lpp, mean_lpp2);
        Means {
            frames: self.frames,
            true_chains_per_frame: self.sum_ntrue / self.frames.max(1) as f64,
            mean_n,
            mean_z,
            mean_ree: mean_ree2.sqrt(),
            mean_lpp,
            ne_ck: estimators.ne_ck,
            ne_mk: estimators.ne_mk,
            ne_cc: estimators.ne_cc,
            ne_mc: estimators.ne_mc,
        }
    }

    /// Flush per-chain files and write the summary file.
    pub fn finalize(&mut self) -> Result<Means> {
        self.z_file.flush()?;
        self.lpp_file.flush()?;
        let m = self.means();

        let path = format!("{}_summary.dat", self.prefix);
        let mut f =
            BufWriter::new(File::create(&path).with_context(|| format!("creating {path}"))?);
        writeln!(f, "# entangl_rs summary")?;
        writeln!(f, "# frames                 {}", m.frames)?;
        writeln!(f, "# true_chains_per_frame  {:.3}", m.true_chains_per_frame)?;
        writeln!(f, "# <N>                    {:.5}", m.mean_n)?;
        writeln!(f, "# <Ree>                  {:.5}", m.mean_ree)?;
        writeln!(f, "# <Lpp>                  {:.5}", m.mean_lpp)?;
        writeln!(f, "# <Z>                    {:.5}", m.mean_z)?;
        writeln!(f, "# Ne_CK (classical kink) {:.5}", m.ne_ck)?;
        writeln!(f, "# Ne_MK (modified kink)  {:.5}", m.ne_mk)?;
        writeln!(f, "# Ne_CC (classical coil) {:.5}", m.ne_cc)?;
        writeln!(f, "# Ne_MC (modified coil)  {:.5}", m.ne_mc)?;
        f.flush()?;
        Ok(m)
    }
}

/// Aggregated means for reporting.
#[derive(Clone, Copy, Debug)]
pub struct Means {
    pub frames: usize,
    pub true_chains_per_frame: f64,
    pub mean_n: f64,
    pub mean_z: f64,
    pub mean_ree: f64,
    pub mean_lpp: f64,
    pub ne_ck: f64,
    pub ne_mk: f64,
    pub ne_cc: f64,
    pub ne_mc: f64,
}

impl Means {
    /// Human-readable one-block report.
    pub fn report(&self, path_prefix: &Path) -> String {
        format!(
            "results ({} frames, {:.1} true chains/frame):\n  <N>   = {:.4}\n  <Ree> = {:.4}\n  <Lpp> = {:.4}\n  <Z>   = {:.4}\n  Ne_CK = {:.4}   Ne_MK = {:.4}   Ne_CC = {:.4}   Ne_MC = {:.4}\n  per-chain data: {}_Z_values.dat, {}_Lpp_values.dat",
            self.frames,
            self.true_chains_per_frame,
            self.mean_n,
            self.mean_ree,
            self.mean_lpp,
            self.mean_z,
            self.ne_ck,
            self.ne_mk,
            self.ne_cc,
            self.ne_mc,
            path_prefix.display(),
            path_prefix.display(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimators_match_z1plus_definitions() {
        let n = 11.0;
        let z = 13.0;
        let ree2 = 0.86_f64.powi(2);
        let lpp = 6.83;
        let e = z1_estimators(n, z, ree2, lpp, lpp.powi(2));

        assert!((e.ne_ck - 110.0 / 141.0).abs() < 1.0e-12);
        assert!((e.ne_mk - 11.0 / 13.0).abs() < 1.0e-12);
        assert!((e.ne_cc - 10.0 * ree2 / lpp.powi(2)).abs() < 1.0e-12);
        assert!((e.ne_mc - 10.0 * ree2 / (lpp.powi(2) - ree2)).abs() < 1.0e-12);
    }

    #[test]
    fn classical_kink_matches_benchmark_10() {
        let e = z1_estimators(200.0, 4.98, 1.0, 1.0, 2.0);
        assert!((e.ne_ck - 33.416_735_235_344).abs() < 1.0e-8);
        assert!((e.ne_mk - 40.160_642_57).abs() < 1.0e-8);
    }
}
