use entangl_rs::chain::{build_chain_indices, frame_from_state, Grouping};
use molar::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

const FILES: [&str; 2] = [
    "rouse_polymers_20chains_20bonds_unwrapped.data",
    "rouse_polymers_20chains_20bonds_wrapped_images.data",
];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name)
}

#[test]
fn lammps_topology_and_periodic_coordinates() {
    let mut frames = Vec::new();
    for name in FILES {
        let sys = System::from_file(fixture(name)).unwrap();
        assert_eq!(sys.select_all().len(), 420);
        assert_eq!(sys.topology().bonds.len(), 400);
        assert_eq!(sys.topology().molecules.len(), 20);
        let ext = sys.state().get_box().unwrap().get_box_extents();
        assert!((ext - Vector3f::new(40.0, 40.0, 40.0)).norm() < 1e-5);

        let expected: Vec<Vec<usize>> = (0..20).map(|c| (c * 21..(c + 1) * 21).collect()).collect();
        for group in [
            Grouping::Auto,
            Grouping::Bonds,
            Grouping::Molecule,
            Grouping::Residue,
        ] {
            assert_eq!(build_chain_indices(&sys, "all", group).unwrap(), expected);
        }
        let expected_bonds: Vec<_> = expected
            .iter()
            .flat_map(|c| c.windows(2).map(|w| [w[0], w[1]]))
            .collect();
        assert_eq!(
            sys.topology()
                .bonds
                .iter()
                .map(|b| b.pair())
                .collect::<Vec<_>>(),
            expected_bonds
        );

        // Independently reconstruct file coordinates: undo wrapping with image
        // flags and translate the [-20,20] box to molar's zero origin.
        let text = std::fs::read_to_string(fixture(name)).unwrap();
        let mut atoms = text
            .lines()
            .skip_while(|l| !l.starts_with("Atoms"))
            .skip(1)
            .filter(|l| !l.trim().is_empty());
        for i in 0..420 {
            let fields: Vec<_> = atoms.next().unwrap().split_whitespace().collect();
            assert_eq!(fields[0].parse::<usize>().unwrap(), i + 1);
            let p = sys.state().get_pos(i).unwrap();
            for axis in 0..3 {
                let raw: Float = fields[3 + axis].parse().unwrap();
                let image: Float = if fields.len() == 9 {
                    fields[6 + axis].parse().unwrap()
                } else {
                    0.0
                };
                assert!((p[axis] - (raw + 20.0 + 40.0 * image)).abs() < 1e-5);
            }
        }
        frames.push(frame_from_state(sys.state(), &expected));
    }
    for (a, b) in frames[0].iter().flatten().zip(frames[1].iter().flatten()) {
        assert!(
            (a - b).norm() < 1e-4,
            "wrapped and unwrapped positions differ"
        );
    }
}

#[test]
fn lammps_cli_analysis_agrees_for_both_representations() {
    let dir = std::env::temp_dir().join(format!("entangl-lammps-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut results = Vec::new();
    for (i, name) in FILES.iter().enumerate() {
        let prefix = dir.join(format!("run-{i}"));
        let output = Command::new(env!("CARGO_BIN_EXE_entangl_rs"))
            .arg("-f")
            .arg(fixture(name))
            .arg("--use_struct_file")
            .arg("-o")
            .arg(&prefix)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let read_values = |suffix: &str| {
            std::fs::read_to_string(format!("{}{suffix}", prefix.display()))
                .unwrap()
                .lines()
                .filter(|l| !l.starts_with('#'))
                .flat_map(|l| l.split_whitespace())
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let z = read_values("_Z_values.dat");
        let lpp = read_values("_Lpp_values.dat");
        assert_eq!(z.len(), 20);
        assert_eq!(lpp.len(), 20);
        assert!(z.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(lpp.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(
            std::fs::metadata(format!("{}_summary.dat", prefix.display()))
                .unwrap()
                .len()
                > 0
        );
        results.push((z, lpp));
    }
    assert_eq!(results[0].0, results[1].0);
    for (a, b) in results[0].1.iter().zip(&results[1].1) {
        assert!((a - b).abs() < 1e-4);
    }
    std::fs::remove_dir_all(dir).unwrap();
}
