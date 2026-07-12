#!/usr/bin/env python3
"""Compare entangl_rs with the bundled Z1+ oracle on official benchmarks.

The script intentionally has no third-party dependencies.  It extracts the
user-supplied Z1+ distribution into a temporary directory, runs both programs,
and reports aggregate and per-chain errors.  Nothing from the Z1+ distribution
is copied into the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DEFAULT_ARCHIVE = PROJECT_ROOT / "Z1+.zip"
DEFAULT_BINARY = ROOT / "target" / "release" / "z1run"

CHAIN_RE = re.compile(
    r"chain\s+(\d+):\s+N=\s*(\d+)\s+Ree=\s*([+\-\d.eE]+)\s+"
    r"Lpp=\s*([+\-\d.eE]+)\s+Z=(\d+)"
)
SUMMARY_RE = re.compile(
    r"<N>=([+\-\d.eE]+).*?<Ree>=([+\-\d.eE]+).*?"
    r"<Lpp>=([+\-\d.eE]+).*?<Z>=([+\-\d.eE]+)"
)


@dataclass(frozen=True)
class ChainValues:
    n: int
    ree: float
    lpp: float
    z: int


@dataclass(frozen=True)
class Metrics:
    case: str
    chains: int
    oracle_lpp: float
    rust_lpp: float
    lpp_relative_error: float
    lpp_mean_absolute_error: float
    lpp_correlation: float
    oracle_z: float
    rust_z: float
    z_error: float
    z_mean_absolute_error: float
    z_equal_chains: int
    z_over_chains: int
    z_under_chains: int


def parse_cases(spec: str) -> list[str]:
    cases: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            cases.extend(f"{n:02d}" for n in range(lo, hi + 1))
        else:
            cases.append(f"{int(part):02d}")
    return list(dict.fromkeys(cases))


def run(command: list[str], cwd: Path, timeout: float) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return completed.stdout


def extract_distribution(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as zf:
        wanted = {
            "Z1+/Z1+.ex",
            *(f"Z1+/.benchmark-{n:02d}.Z1" for n in range(1, 15)),
        }
        missing = wanted.difference(zf.namelist())
        if missing:
            raise RuntimeError(f"Z1+ archive is missing: {sorted(missing)}")
        for name in wanted:
            zf.extract(name, destination)
    executable = destination / "Z1+" / "Z1+.ex"
    executable.chmod(executable.stat().st_mode | 0o111)
    return destination / "Z1+"


def oracle_parameters() -> str:
    return """&parameters
visualize = .False.
movie = .False.
self_entanglement = .False.
benchmarks = .False.
stats = .False.
logfile = .False.
basic_SP = .False.
lmax_factor = 1.0
thickness = 0.002
user = 'CPC'
debug = .False.
calc_PPA = .False.
calc_CPPA = .False.
/
"""


def read_scalar_lines(path: Path, cast: type[float] | type[int]) -> list[float] | list[int]:
    return [cast(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def run_oracle(distribution: Path, case: str, work: Path, timeout: float) -> tuple[list[ChainValues], tuple[float, float]]:
    case_dir = work / f"oracle-{case}"
    case_dir.mkdir()
    shutil.copy2(distribution / "Z1+.ex", case_dir / "Z1+.ex")
    shutil.copy2(distribution / f".benchmark-{case}.Z1", case_dir / "config.Z1")
    (case_dir / "Z1+parameters").write_text(oracle_parameters())
    run([str(case_dir / "Z1+.ex")], case_dir, timeout)

    n_values = read_scalar_lines(case_dir / "N_values.dat", int)
    ree_values = read_scalar_lines(case_dir / "Ree_values.dat", float)
    lpp_values = read_scalar_lines(case_dir / "Lpp_values.dat", float)
    z_values = read_scalar_lines(case_dir / "Z_values.dat", int)
    lengths = {len(n_values), len(ree_values), len(lpp_values), len(z_values)}
    if len(lengths) != 1:
        raise RuntimeError(f"oracle emitted inconsistent per-chain lengths for benchmark {case}")
    chains = [
        ChainValues(n=n, ree=ree, lpp=lpp, z=z)
        for n, ree, lpp, z in zip(n_values, ree_values, lpp_values, z_values)
    ]
    summary = (case_dir / "Z1+summary.dat").read_text().split()
    return chains, (float(summary[4]), float(summary[5]))


def run_rust(binary: Path, config: Path, extra_args: list[str], timeout: float) -> tuple[list[ChainValues], tuple[float, float]]:
    output = run([str(binary), "--dump", *extra_args, str(config)], ROOT, timeout)
    chains = [
        ChainValues(n=int(n), ree=float(ree), lpp=float(lpp), z=int(z))
        for _, n, ree, lpp, z in CHAIN_RE.findall(output)
        if int(n) >= 3
    ]
    match = SUMMARY_RE.search(output)
    if match is None:
        raise RuntimeError(f"could not parse z1run summary:\n{output}")
    _, _, lpp, z = (float(value) for value in match.groups())
    return chains, (lpp, z)


def correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 1.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denom if denom else 1.0


def compare(case: str, oracle: list[ChainValues], rust: list[ChainValues], oracle_summary: tuple[float, float], rust_summary: tuple[float, float]) -> Metrics:
    if len(oracle) != len(rust):
        raise RuntimeError(
            f"benchmark {case}: oracle has {len(oracle)} true chains, Rust has {len(rust)}"
        )
    oracle_lpps = [chain.lpp for chain in oracle]
    rust_lpps = [chain.lpp for chain in rust]
    z_delta = [actual.z - expected.z for expected, actual in zip(oracle, rust)]
    oracle_lpp, oracle_z = oracle_summary
    rust_lpp, rust_z = rust_summary
    return Metrics(
        case=case,
        chains=len(oracle),
        oracle_lpp=oracle_lpp,
        rust_lpp=rust_lpp,
        lpp_relative_error=(rust_lpp - oracle_lpp) / oracle_lpp,
        lpp_mean_absolute_error=statistics.fmean(
            abs(actual - expected) for expected, actual in zip(oracle_lpps, rust_lpps)
        ),
        lpp_correlation=correlation(oracle_lpps, rust_lpps),
        oracle_z=oracle_z,
        rust_z=rust_z,
        z_error=rust_z - oracle_z,
        z_mean_absolute_error=statistics.fmean(abs(delta) for delta in z_delta),
        z_equal_chains=sum(delta == 0 for delta in z_delta),
        z_over_chains=sum(delta > 0 for delta in z_delta),
        z_under_chains=sum(delta < 0 for delta in z_delta),
    )


def print_metrics(metric: Metrics) -> None:
    print(
        f"{metric.case}: "
        f"Lpp {metric.rust_lpp:.5f}/{metric.oracle_lpp:.5f} "
        f"({metric.lpp_relative_error:+.2%}, MAE {metric.lpp_mean_absolute_error:.4f}, "
        f"r {metric.lpp_correlation:.4f}); "
        f"Z {metric.rust_z:.5f}/{metric.oracle_z:.5f} "
        f"({metric.z_error:+.5f}, MAE {metric.z_mean_absolute_error:.4f}, "
        f"equal/over/under {metric.z_equal_chains}/{metric.z_over_chains}/{metric.z_under_chains})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--cases", default="01-05", help="comma-separated cases/ranges")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per process")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--json", type=Path, help="also write machine-readable metrics")
    parser.add_argument(
        "--rust-arg",
        action="append",
        default=[],
        help="extra z1run argument; repeat for multiple arguments",
    )
    args = parser.parse_args()

    if not args.archive.is_file():
        parser.error(f"Z1+ archive not found: {args.archive}")
    if not args.no_build:
        run(["cargo", "build", "--release", "--offline", "--bin", "z1run"], ROOT, args.timeout)
    if not args.binary.is_file():
        parser.error(f"z1run binary not found: {args.binary}")

    metrics: list[Metrics] = []
    with tempfile.TemporaryDirectory(prefix="entangl-regression-") as tmp_name:
        tmp = Path(tmp_name)
        distribution = extract_distribution(args.archive, tmp / "distribution")
        for case in parse_cases(args.cases):
            config = distribution / f".benchmark-{case}.Z1"
            oracle, oracle_summary = run_oracle(distribution, case, tmp, args.timeout)
            rust, rust_summary = run_rust(args.binary.resolve(), config, args.rust_arg, args.timeout)
            metric = compare(case, oracle, rust, oracle_summary, rust_summary)
            metrics.append(metric)
            print_metrics(metric)

    if args.json:
        args.json.write_text(json.dumps([asdict(metric) for metric in metrics], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(error.stdout or str(error), file=sys.stderr)
        raise SystemExit(error.returncode)
    except subprocess.TimeoutExpired as error:
        print(f"command timed out after {error.timeout}s: {error.cmd}", file=sys.stderr)
        raise SystemExit(124)
