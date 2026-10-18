"""Command line: the commands run and give the same answer as the Python API; exit codes 0 / 1 / 2."""

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from panda_md import FitSpec, ProfileSet, SlitPore
from panda_md.analysis.blocks import fit_blocks
from panda_md.analysis.compare import compare_shapes
from panda_md.cli import main
from panda_md.core.enums import NoiseModel, ProfileSampling, Regime
from panda_md.core.errors import PandaWarning
from panda_md.core.fitting import fit
from panda_md.core.profile import DensityProfile
from panda_md.synthetic.profiles import synthetic_profile

PORE = SlitPore(width_nm=8.0, length_nm=16.0)
TRUTH = {"theta": 120.0, "delta": 0.03, "phi": 0.4}
DENSITY_BULK = 30.0
MD_DATA = Path(__file__).parents[1] / "md" / "data"
RHO = ("--rho-bulk", str(DENSITY_BULK))
WATER_DENSITY_BULK = 33.13  # ρ_bulk of water oxygens in the graphite system, as in tests/md
WATER_RHO = ("--rho-bulk", str(WATER_DENSITY_BULK))


def run_cli(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, str, str]:
    code = main(list(arguments))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture(scope="module")
def roll_profile() -> DensityProfile:
    return synthetic_profile(
        PORE.interface("roll"),
        TRUTH,
        density_bulk_per_nm3=DENSITY_BULK,
        noise_std=0.005,
        seed=1,
        noise_model=NoiseModel.POISSON,
        sampling=ProfileSampling.BIN_AVERAGE,
    )


@pytest.fixture
def profile_file(tmp_path: Path, roll_profile: DensityProfile) -> Path:
    path = tmp_path / "profile.npz"
    roll_profile.save(path)
    return path


@pytest.fixture
def pore_file(tmp_path: Path) -> Path:
    path = tmp_path / "pore.json"
    path.write_text(PORE.model_dump_json())
    return path


def read_values(path: Path) -> dict[str, float]:
    values: dict[str, float] = json.loads(path.read_text())["fit"]["values"]
    return values


# --- panda shapes ---


def test_shapes_lists_every_pore_and_its_shapes(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run_cli(capsys, "shapes")
    assert code == 0
    for pore_name, shape_name in (("slit", "roll"), ("square", "filament_w1c0"), ("triangle", "filament_w1c0")):
        assert pore_name in out
        assert shape_name in out


def test_shapes_for_one_pore_shows_parameters_bounds_regimes_and_axis(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run_cli(capsys, "shapes", "--pore", "slit")
    assert code == 0
    assert "width_nm" in out  # pore fields — what is set with -g
    assert "theta" in out and "[0, 180]" in out
    assert "non_wetting" in out
    assert "horizontal" in out  # roll axis
    assert "square" not in out


def test_shapes_unknown_pore_is_input_error(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run_cli(capsys, "shapes", "--pore", "hexagon")
    assert code == 1
    assert "hexagon" in err and "slit" in err


# --- panda fit ---


def test_fit_matches_python_api(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, profile_file: Path, pore_file: Path
) -> None:
    output = tmp_path / "result.json"
    code, out, _ = run_cli(
        capsys, "fit", str(profile_file), "--pore-file", str(pore_file), "--shape", "roll", *RHO, "-o", str(output)
    )
    expected = fit(
        PORE.interface("roll"), DensityProfile.load(profile_file), FitSpec(density_bulk_per_nm3=DENSITY_BULK)
    )
    assert code == 0
    assert read_values(output) == pytest.approx(expected.values, abs=1e-9)
    assert "theta" in out


def test_fit_parameter_options_build_the_same_spec(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, profile_file: Path
) -> None:
    output = tmp_path / "result.json"
    code, _, _ = run_cli(
        capsys,
        "fit", str(profile_file),
        "--pore", "slit", "-g", "width_nm=8", "-g", "length_nm=16",
        "--shape", "roll",
        "--optimize", "theta", "--optimize", "delta",
        "--fix", "phi=0.4",
        "--bounds", "theta=100:170",
        "--rho-bulk", "30",
        "--regime", "non_wetting",
        "-o", str(output),
    )  # fmt: skip
    spec = FitSpec(
        optimize=("theta", "delta"),
        fixed={"phi": 0.4},
        bounds={"theta": (100.0, 170.0)},
        density_bulk_per_nm3=30.0,
        regime=Regime.NON_WETTING,
    )
    expected = fit(PORE.interface("roll"), DensityProfile.load(profile_file), spec)
    assert code == 0
    assert read_values(output) == pytest.approx(expected.values, abs=1e-9)
    saved = json.loads(output.read_text())
    assert saved["fit"]["spec"]["fixed"] == {"phi": 0.4}
    assert saved["blocks"] is None and saved["comparison"] is None


def test_fit_with_problem_flags_exits_with_2(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, profile_file: Path, pore_file: Path
) -> None:
    code, out, _ = run_cli(
        capsys,
        "fit",
        str(profile_file),
        "--pore-file",
        str(pore_file),
        "--shape",
        "roll",
        *RHO,
        "--bounds",
        "theta=95:105",
    )
    assert code == 2
    assert "at bound: theta" in out


def test_fit_auto_compares_shapes_and_keeps_the_best(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, profile_file: Path, pore_file: Path
) -> None:
    output = tmp_path / "result.json"
    code, out, err = run_cli(
        capsys, "fit", str(profile_file), "--pore-file", str(pore_file), "--shape", "auto", *RHO, "-o", str(output)
    )
    with pytest.warns(PandaWarning, match="indistinguishable"):  # the profile cannot tell roll from perforation
        comparison = compare_shapes(PORE, DensityProfile.load(profile_file), FitSpec(density_bulk_per_nm3=DENSITY_BULK))
    assert code in (0, 2)
    assert "shape comparison" in out
    assert "indistinguishable" in err
    saved = json.loads(output.read_text())
    assert saved["fit"]["shape_name"] == comparison.best.shape_name == "roll"
    assert saved["fit"]["values"] == pytest.approx(comparison.best.values, abs=1e-9)
    assert [row["shape_name"] for row in saved["comparison"]["rows"]] == [row.shape_name for row in comparison.rows]


def test_fit_profile_set_uses_mean_and_blocks_fit_each_block(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    profiles_path = MD_DATA / "graphite_water_profiles.npz"
    pore_path = MD_DATA / "graphite_pore.json"
    pore = SlitPore.model_validate_json(pore_path.read_text())
    profiles = ProfileSet.load(profiles_path)

    output = tmp_path / "mean.json"
    code, _, _ = run_cli(
        capsys,
        "fit",
        str(profiles_path),
        "--pore-file",
        str(pore_path),
        "--shape",
        "roll",
        *WATER_RHO,
        "-o",
        str(output),
    )
    spec = FitSpec(density_bulk_per_nm3=WATER_DENSITY_BULK)
    expected = fit(pore.interface("roll"), profiles.mean(), spec)
    assert code in (0, 2)
    assert read_values(output) == pytest.approx(expected.values, abs=1e-9)

    blocks_output = tmp_path / "blocks.json"
    code, out, _ = run_cli(
        capsys, "fit", str(profiles_path), "--pore-file", str(pore_path), "--shape", "roll", *WATER_RHO, "--blocks",
        "-o", str(blocks_output),
    )  # fmt: skip
    expected_blocks = fit_blocks(pore.interface("roll"), profiles, spec)
    assert code in (0, 2)
    assert "mean ± standard error" in out
    saved = json.loads(blocks_output.read_text())
    assert saved["fit"]["values"] == pytest.approx(expected.values, abs=1e-9)  # and the fit of the mean profile
    assert [result["values"]["theta"] for result in saved["blocks"]["results"]] == pytest.approx(
        list(expected_blocks.values("theta")), abs=1e-9
    )


def test_fit_csv_profile_without_metadata(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, roll_profile: DensityProfile, pore_file: Path
) -> None:
    path = tmp_path / "profile.csv"
    np.savetxt(path, np.column_stack([roll_profile.coordinate_nm, roll_profile.density_per_nm3]), delimiter=",",
               header="coordinate_nm,density_per_nm3", comments="# ")  # fmt: skip
    output = tmp_path / "result.json"
    code, _, err = run_cli(
        capsys,
        "fit",
        str(path),
        "--pore-file",
        str(pore_file),
        "--shape",
        "roll",
        "--rho-bulk",
        "30",
        "-o",
        str(output),
    )
    expected = fit(PORE.interface("roll"), DensityProfile.load(path), FitSpec(density_bulk_per_nm3=30.0))
    assert code == 0, err
    assert read_values(output) == pytest.approx(expected.values, abs=1e-9)


def test_fit_plot_writes_figure(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, profile_file: Path, pore_file: Path
) -> None:
    pytest.importorskip("matplotlib")
    figure = tmp_path / "fit.png"
    code, _, _ = run_cli(
        capsys, "fit", str(profile_file), "--pore-file", str(pore_file), "--shape", "roll", *RHO, "--plot", str(figure)
    )
    assert code == 0
    assert figure.stat().st_size > 0


def test_fit_plot_without_matplotlib_fails_before_fitting(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile_file: Path,
    pore_file: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # import matplotlib → ImportError
    monkeypatch.delitem(sys.modules, "panda_md.plotting", raising=False)
    monkeypatch.delitem(sys.modules, "panda_md.plotting.figures", raising=False)
    output = tmp_path / "result.json"
    code, out, err = run_cli(
        capsys, "fit", str(profile_file), "--pore-file", str(pore_file), "--shape", "roll", *RHO,
        "--plot", str(tmp_path / "fit.png"), "-o", str(output),
    )  # fmt: skip
    assert code == 1
    assert "matplotlib" in err
    assert out == "" and not output.exists()  # the fit did not run


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--fix", "phi"], "expected name=value"),
        (["--fix", "gamma=1"], "gamma"),
        (["--bounds", "theta=100"], "expected name=lower:upper"),
        (["--fix", "phi=0.4", "--optimize", "phi"], "phi"),
        (["--shape", "spiral"], "spiral"),
        (["--regime", "sticky"], "sticky"),
    ],
)
def test_fit_bad_options_are_input_errors_without_traceback(
    capsys: pytest.CaptureFixture[str], profile_file: Path, pore_file: Path, arguments: list[str], message: str
) -> None:
    base = ["fit", str(profile_file), "--pore-file", str(pore_file), *RHO]
    if "--shape" not in arguments:
        base += ["--shape", "roll"]
    code, _, err = run_cli(capsys, *base, *arguments)
    assert code == 1
    assert message in err
    assert "Traceback" not in err


def test_fit_missing_file_and_missing_pore_are_input_errors(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, profile_file: Path
) -> None:
    code, _, err = run_cli(
        capsys, "fit", str(tmp_path / "absent.npz"), "--pore", "slit", "-g", "width_nm=8", "-g", "length_nm=16"
    )
    assert code == 1 and "absent.npz" in err
    code, _, err = run_cli(capsys, "fit", str(profile_file), "--shape", "roll")
    assert code == 1 and "--pore" in err
    code, _, err = run_cli(capsys, "fit", str(profile_file), "--pore", "slit", "-g", "width_nm=8", "--shape", "roll")
    assert code == 1 and "length_nm" in err


def test_usage_errors_exit_with_1_not_2(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run_cli(capsys, "fit")  # a required argument is missing: click gives code 2, ours is 2 for fit flags
    assert code == 1
    assert err


# --- panda profile ---


def write_md_files(directory: Path) -> tuple[Path, Path]:
    """Synthetic slit trajectory in GROMACS format (.gro + .xtc): liquid LIQ, walls WAL."""
    mda = pytest.importorskip("MDAnalysis")
    from MDAnalysis.coordinates.memory import MemoryReader

    from panda_md.synthetic.points import synthetic_trajectory

    trajectory = synthetic_trajectory(PORE.interface("roll"), TRUTH, density_bulk_per_nm3=5.0, frame_count=4, seed=3)
    liquid = trajectory.groups["liquid"]
    walls = trajectory.groups["walls"]
    atom_count = trajectory.positions_nm.shape[1]
    universe = mda.Universe.empty(
        atom_count, n_residues=atom_count, atom_resindex=np.arange(atom_count), trajectory=True
    )
    resnames = np.empty(atom_count, dtype=object)
    resnames[liquid] = "LIQ"
    resnames[walls] = "WAL"
    universe.add_TopologyAttr("resname", list(resnames))
    names = np.empty(atom_count, dtype=object)
    names[liquid] = "O"  # known elements: MDAnalysis guesses the masses without warnings
    names[walls] = "C"
    universe.add_TopologyAttr("name", list(names))
    universe.add_TopologyAttr("resid", list(range(1, atom_count + 1)))
    dimensions = np.column_stack([trajectory.boxes_nm * 10.0, np.full((len(trajectory.boxes_nm), 3), 90.0)])
    universe.load_new(trajectory.positions_nm * 10.0, format=MemoryReader, dimensions=dimensions, dt=10.0)
    topology = directory / "system.gro"
    coordinates = directory / "traj.xtc"
    universe.atoms.write(str(topology))
    with mda.Writer(str(coordinates), atom_count) as writer:
        for _ in universe.trajectory:
            writer.write(universe.atoms)
    return topology, coordinates


def test_profile_matches_python_api(capsys: pytest.CaptureFixture[str], tmp_path: Path, pore_file: Path) -> None:
    from panda_md import density_profile, load_trajectory

    topology, coordinates = write_md_files(tmp_path)
    output = tmp_path / "profiles.npz"
    code, out, err = run_cli(
        capsys, "profile", str(topology), str(coordinates),
        "--pore-file", str(pore_file), "--walls", "resname WAL", "--select", "resname LIQ",
        "--bins", "80", "--block", "20 ps", "-o", str(output),
    )  # fmt: skip
    assert code == 0, err
    pore = PORE.model_copy(update={"wall_selection": "resname WAL"})
    expected = density_profile(load_trajectory(topology, coordinates), pore, "resname LIQ", bins=80, block="20 ps")
    saved = ProfileSet.load(output)
    np.testing.assert_allclose(saved.block_densities_per_nm3, expected.block_densities_per_nm3)
    np.testing.assert_allclose(saved.bin_edges_nm, expected.bin_edges_nm)
    assert saved.block_count == 2
    assert "2 block(s)" in out


def test_profile_walls_as_two_coordinates_fix_the_pore_center(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, pore_file: Path
) -> None:
    from panda_md import density_profile, load_trajectory

    topology, coordinates = write_md_files(tmp_path)
    output = tmp_path / "profiles.npz"
    code, _, err = run_cli(
        capsys, "profile", str(topology), str(coordinates), "--pore-file", str(pore_file),
        "--walls", "1.0,9.0", "--select", "resname LIQ", "--bin-width", "0.1", "-o", str(output),
    )  # fmt: skip
    assert code == 0, err
    expected = density_profile(
        load_trajectory(topology, coordinates), PORE, "resname LIQ", bin_width_nm=0.1, pore_center_nm=5.0
    )
    np.testing.assert_allclose(ProfileSet.load(output).block_densities_per_nm3, expected.block_densities_per_nm3)


def test_profile_needs_walls_for_pore_centering(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, pore_file: Path
) -> None:
    topology, coordinates = write_md_files(tmp_path)
    base = ["profile", str(topology), str(coordinates), "--pore-file", str(pore_file), "--select", "resname LIQ"]
    code, _, err = run_cli(capsys, *base, "--center-by", "pore", "-o", str(tmp_path / "p.npz"))
    assert code == 1
    assert "wall selection" in err
    code, _, err = run_cli(capsys, *base, "-o", str(tmp_path / "liquid.npz"))  # by default — on the liquid
    assert code == 0
    assert "centred on the liquid" in err
    assert ProfileSet.load(tmp_path / "liquid.npz").centering.value == "liquid"


def test_profile_derives_slit_from_walls_and_saves_pore_for_fit(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from panda_md import density_profile, load_trajectory

    topology, coordinates = write_md_files(tmp_path)
    output = tmp_path / "profiles.npz"
    pore_output = tmp_path / "pore.json"
    code, out, err = run_cli(
        capsys, "profile", str(topology), str(coordinates), "--walls", "resname WAL", "--select", "resname LIQ",
        "--bins", "80", "-o", str(output), "--pore-output", str(pore_output),
    )  # fmt: skip
    assert code == 0, err
    trajectory = load_trajectory(topology, coordinates)
    pore = SlitPore.from_trajectory(trajectory, walls="resname WAL", liquid="resname LIQ")
    expected = density_profile(trajectory, pore, "resname LIQ", bins=80)
    np.testing.assert_allclose(ProfileSet.load(output).block_densities_per_nm3, expected.block_densities_per_nm3)
    assert "width_nm" in out  # the dimensions found from the trajectory are shown to the user

    fit_output = tmp_path / "result.json"
    code, _, err = run_cli(
        capsys, "fit", str(output), "--pore-file", str(pore_output), "--shape", "roll", "--rho-bulk", "5",
        "-o", str(fit_output),
    )  # fmt: skip
    assert code in (0, 2), err
    assert json.loads(fit_output.read_text())["fit"]["pore_name"] == "slit"


def test_profile_partial_geometry_is_input_error(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    topology, coordinates = write_md_files(tmp_path)
    code, _, err = run_cli(
        capsys, "profile", str(topology), str(coordinates), "-g", "width_nm=8", "--walls", "resname WAL",
        "--select", "resname LIQ", "-o", str(tmp_path / "p.npz"),
    )  # fmt: skip
    assert code == 1
    assert "length_nm" in err


# --- panda run ---


def test_run_config_matches_separate_commands(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, profile_file: Path, pore_file: Path
) -> None:
    config = tmp_path / "panda.toml"
    config.write_text(
        "[pore]\n"
        'type = "slit"\n'
        "width_nm = 8.0\n"
        "length_nm = 16.0\n"
        "\n"
        "[profile]\n"
        f'input = "{profile_file.name}"\n'
        "\n"
        "[fit]\n"
        'shape = "roll"\n'
        'optimize = ["theta", "delta"]\n'
        "fixed = { phi = 0.4 }\n"
        "bounds = { theta = [100.0, 170.0] }\n"
        "density_bulk_per_nm3 = 30.0\n"
        'output = "result.json"\n'
    )
    code, out, err = run_cli(capsys, "run", str(config))
    expected = fit(
        PORE.interface("roll"),
        DensityProfile.load(profile_file),
        FitSpec(
            optimize=("theta", "delta"), fixed={"phi": 0.4}, bounds={"theta": (100.0, 170.0)}, density_bulk_per_nm3=30.0
        ),
    )
    assert code == 0, err
    assert read_values(tmp_path / "result.json") == pytest.approx(expected.values, abs=1e-9)  # path from the config
    assert "theta" in out


def test_run_config_builds_profile_from_trajectory(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    topology, coordinates = write_md_files(tmp_path)
    config = tmp_path / "panda.toml"
    config.write_text(
        "[pore]\n"
        'type = "slit"\n'
        "width_nm = 8.0\n"
        "length_nm = 16.0\n"
        'wall_selection = "resname WAL"\n'
        "\n"
        "[profile]\n"
        f'topology = "{topology.name}"\n'
        f'trajectory = ["{coordinates.name}"]\n'
        'select = "resname LIQ"\n'
        "bins = 80\n"
        'output = "profiles.npz"\n'
        "\n"
        "[fit]\n"
        'shape = "roll"\n'
        "density_bulk_per_nm3 = 5.0\n"
    )
    code, out, err = run_cli(capsys, "run", str(config))
    assert code in (0, 2), err
    assert (tmp_path / "profiles.npz").exists()
    assert "roll" in out


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('[pore]\ntype = "slit"\nwidth_nm = 8.0\nlength_nm = 16.0\n[fit]\nshape = "roll"\n', "profile"),
        ('[pore]\ntype = "slit"\nwidth_nm = 8.0\nlength_nm = 16.0\n[profile]\ninput = "p.npz"\nbinz = 3\n', "binz"),
        (
            '[pore]\ntype = "slit"\nwidth_nm = 8.0\nlength_nm = 16.0\n[profile]\ninput = "p.npz"\ntopology = "s.gro"\n',
            "input",
        ),
        ("[pore\n", "TOML"),
    ],
)
def test_run_bad_config_is_input_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, body: str, message: str
) -> None:
    config = tmp_path / "panda.toml"
    config.write_text(body)
    code, _, err = run_cli(capsys, "run", str(config))
    assert code == 1
    assert message in err
    assert "Traceback" not in err


# --- entry point ---


def test_console_entry_point_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "panda_md", "shapes", "--pore", "slit"], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert "roll" in completed.stdout


# --- stage 8 final review ---


def fit_arguments(profile_file: Path, pore_file: Path) -> list[str]:
    return ["fit", str(profile_file), "--pore-file", str(pore_file), "--shape", "roll", *RHO]


def test_npz_that_is_not_a_panda_profile_is_input_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, pore_file: Path, roll_profile: DensityProfile
) -> None:
    foreign = tmp_path / "foreign.npz"
    np.savez(foreign, coordinate_nm=roll_profile.coordinate_nm, density_per_nm3=roll_profile.density_per_nm3)
    renamed = tmp_path / "renamed.npz"
    with renamed.open("wb") as stream:
        np.save(stream, roll_profile.density_per_nm3)
    for path in (foreign, renamed):
        code, _, err = run_cli(capsys, *fit_arguments(path, pore_file))
        assert code == 1
        assert "is not a PANDA profile" in err and path.name in err


@pytest.mark.parametrize("header", [None, "# z,rho"])
def test_csv_from_another_program_uses_column_order(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    pore_file: Path,
    roll_profile: DensityProfile,
    header: str | None,
) -> None:
    path = tmp_path / "profile.csv"
    rows = "\n".join(
        f"{float(z)!r},{float(rho)!r}"
        for z, rho in zip(roll_profile.coordinate_nm, roll_profile.density_per_nm3, strict=True)
    )
    path.write_text((header + "\n" if header else "") + rows + "\n")
    output = tmp_path / "result.json"
    code, _, err = run_cli(capsys, *fit_arguments(path, pore_file), "-o", str(output))
    reference = DensityProfile(coordinate_nm=roll_profile.coordinate_nm, density_per_nm3=roll_profile.density_per_nm3)
    expected = fit(PORE.interface("roll"), reference, FitSpec(density_bulk_per_nm3=DENSITY_BULK))
    assert code == 0, err
    assert read_values(output) == pytest.approx(expected.values, abs=1e-9)


def test_csv_with_wrong_column_count_is_input_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, pore_file: Path
) -> None:
    path = tmp_path / "profile.csv"
    path.write_text("1,2,3,4\n2,3,4,5\n")
    code, _, err = run_cli(capsys, *fit_arguments(path, pore_file))
    assert code == 1
    assert "column(s), expected 2 or 3" in err


def test_bad_outputs_are_rejected_before_any_work(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, profile_file: Path, pore_file: Path
) -> None:
    code, out, err = run_cli(capsys, *fit_arguments(profile_file, pore_file), "-o", str(tmp_path / "absent" / "r.json"))
    assert code == 1 and "absent" in err and out == ""
    pytest.importorskip("matplotlib")
    code, out, err = run_cli(capsys, *fit_arguments(profile_file, pore_file), "--plot", str(tmp_path / "fit.xyz"))
    assert code == 1 and "xyz" in err and out == ""


def test_profile_output_must_be_npz_before_reading_trajectory(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, pore_file: Path
) -> None:
    topology, coordinates = write_md_files(tmp_path)
    code, out, err = run_cli(
        capsys, "profile", str(topology), str(coordinates), "--pore-file", str(pore_file), "--walls", "resname WAL",
        "--select", "resname LIQ", "-o", str(tmp_path / "profiles.csv"),
    )  # fmt: skip
    assert code == 1 and ".npz" in err and out == ""


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--bins", "80", "--bin-width", "0.1"], "--bins"),
        (["--bin-width", "-1"], "--bin-width"),
        (["-g", "base_aspect_ratio=0.5"], "base_aspect_ratio"),  # no dimensions, but with a field that is not used
        (["--walls", "1.0,9.0"], "wall coordinates"),  # wall coordinates do not measure the slit
    ],
)
def test_profile_contradictory_options_are_input_errors(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, arguments: list[str], message: str
) -> None:
    topology, coordinates = write_md_files(tmp_path)
    base = ["profile", str(topology), str(coordinates), "--select", "resname LIQ", "-o", str(tmp_path / "p.npz")]
    if "--walls" not in arguments:
        base += ["--walls", "resname WAL"]
    code, _, err = run_cli(capsys, *base, *arguments)
    assert code == 1
    assert message in err


def test_fit_empty_optimize_is_input_error(
    capsys: pytest.CaptureFixture[str], profile_file: Path, pore_file: Path
) -> None:
    code, out, err = run_cli(capsys, *fit_arguments(profile_file, pore_file), "--optimize", ",")
    assert code == 1 and "--optimize" in err and out == ""


def test_validation_errors_are_short_and_name_the_field(capsys: pytest.CaptureFixture[str], profile_file: Path) -> None:
    code, _, err = run_cli(capsys, "fit", str(profile_file), "-g", "width_nm=abc", "-g", "length_nm=16")
    assert code == 1
    assert "width_nm" in err
    assert "pydantic.dev" not in err


def test_no_arguments_prints_help_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run_cli(capsys)
    assert code == 0
    assert "fit" in out
    assert "error" not in err


def test_help_keeps_square_brackets(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run_cli(capsys, "profile", "--help")
    assert code == 0
    assert "panda-md[md]" in re.sub(r"-\s+", "-", out)  # the help formatter may wrap the line after "panda-"


def test_shapes_lists_the_profile_shift(capsys: pytest.CaptureFixture[str]) -> None:
    _, out, _ = run_cli(capsys, "shapes", "--pore", "slit")
    assert "shift" in out


def test_interrupt_exits_with_130(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, profile_file: Path, pore_file: Path
) -> None:
    cli_app = sys.modules["panda_md.cli.app"]  # the name panda_md.cli.app is taken by the Typer object

    def interrupt(*_: object, **__: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_app, "run_fit", interrupt)
    code, _, _ = run_cli(capsys, *fit_arguments(profile_file, pore_file))
    assert code == 130


def test_importing_main_module_does_not_run_the_command() -> None:
    importlib.import_module("panda_md.__main__")  # no SystemExit


def write_config(directory: Path, pore: str, profile: str, fit_section: str | None) -> Path:
    config = directory / "panda.toml"
    text = f"[pore]\n{pore}\n[profile]\n{profile}\n"
    if fit_section is not None:
        text += f"[fit]\n{fit_section}\n"
    config.write_text(text)
    return config


def test_run_readme_variant_measures_slit_and_resolves_every_path(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    pytest.importorskip("matplotlib")
    topology, coordinates = write_md_files(tmp_path)
    config = write_config(
        tmp_path,
        'type = "slit"\nwall_selection = "resname WAL"',
        f'topology = "{topology.name}"\ntrajectory = ["{coordinates.name}"]\nselect = "resname LIQ"\nbins = 80\n'
        'block = "20 ps"\noutput = "profiles.npz"\npore_output = "pore.json"',
        'shape = "roll"\ndensity_bulk_per_nm3 = 5.0\nblocks = true\noutput = "result.json"\nplot = "fit.png"',
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    previous = Path.cwd()
    try:
        import os

        os.chdir(elsewhere)  # paths are relative to the settings file, not to the current directory
        code, _, err = run_cli(capsys, "run", str(config))
    finally:
        os.chdir(previous)
    assert code in (0, 2), err
    for name in ("profiles.npz", "pore.json", "result.json", "fit.png"):
        assert (tmp_path / name).exists(), name
    assert json.loads((tmp_path / "pore.json").read_text())["type"] == "slit"


def test_run_checks_shape_and_plot_before_reading_trajectory(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    topology, coordinates = write_md_files(tmp_path)
    profile = (
        f'topology = "{topology.name}"\ntrajectory = ["{coordinates.name}"]\nselect = "resname LIQ"\n'
        'output = "profiles.npz"'
    )
    pore = 'type = "slit"\nwall_selection = "resname WAL"'
    config = write_config(tmp_path, pore, profile, 'shape = "rol"')
    code, out, err = run_cli(capsys, "run", str(config))
    assert code == 1 and "rol" in err and out == ""
    assert not (tmp_path / "profiles.npz").exists()

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "panda_md.plotting", raising=False)
    monkeypatch.delitem(sys.modules, "panda_md.plotting.figures", raising=False)
    config = write_config(tmp_path, pore, profile, 'shape = "roll"\nplot = "fit.png"')
    code, out, err = run_cli(capsys, "run", str(config))
    assert code == 1 and "matplotlib" in err and out == ""
    assert not (tmp_path / "profiles.npz").exists()


@pytest.mark.parametrize(
    ("profile", "fit_section", "message"),
    [
        ('input = "p.npz"', 'shape = "roll"', "[pore]"),  # the hint is the file section, not the command options
        ('input = "p.npz"', None, "[fit]"),  # a ready profile and no fit — nothing to do
        ('input = "p.npz"\nbins = 80\nbin_width_nm = 0.1', 'shape = "roll"', "bin_width_nm"),
    ],
)
def test_run_config_mistakes_are_explained(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, profile: str, fit_section: str | None, message: str
) -> None:
    config = write_config(tmp_path, 'type = "slit"', profile, fit_section)
    code, _, err = run_cli(capsys, "run", str(config))
    assert code == 1
    assert message in err
