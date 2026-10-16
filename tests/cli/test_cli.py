"""Командная строка: команды запускаются и дают тот же ответ, что Python API; коды возврата 0 / 1 / 2."""

import json
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
from panda_md.core.fitting import fit
from panda_md.core.profile import DensityProfile
from panda_md.synthetic.profiles import synthetic_profile

PORE = SlitPore(width_nm=8.0, length_nm=16.0)
TRUTH = {"theta": 120.0, "delta": 0.03, "phi": 0.4}
DENSITY_BULK = 30.0
MD_DATA = Path(__file__).parents[1] / "md" / "data"
RHO = ("--rho-bulk", str(DENSITY_BULK))
WATER_DENSITY_BULK = 33.13  # ρ_bulk кислородов воды в графитовой системе, как в tests/md
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
    values: dict[str, float] = json.loads(path.read_text())["values"]
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
    assert "width_nm" in out  # поля поры — то, что задаётся через -g
    assert "theta" in out and "[0, 180]" in out
    assert "non_wetting" in out
    assert "horizontal" in out  # ось валика
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
    assert json.loads(output.read_text())["spec"]["fixed"] == {"phi": 0.4}


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
    assert "у границы" in out


def test_fit_auto_compares_shapes_and_keeps_the_best(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, profile_file: Path, pore_file: Path
) -> None:
    output = tmp_path / "result.json"
    code, out, _ = run_cli(
        capsys, "fit", str(profile_file), "--pore-file", str(pore_file), "--shape", "auto", *RHO, "-o", str(output)
    )
    comparison = compare_shapes(PORE, DensityProfile.load(profile_file), FitSpec(density_bulk_per_nm3=DENSITY_BULK))
    assert code in (0, 2)
    assert "сравнение форм" in out
    saved = json.loads(output.read_text())
    assert saved["shape_name"] == comparison.best.shape_name == "roll"
    assert saved["values"] == pytest.approx(comparison.best.values, abs=1e-9)


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
    assert "среднее ± ошибка среднего" in out
    saved = json.loads(blocks_output.read_text())
    assert [result["values"]["theta"] for result in saved["results"]] == pytest.approx(
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
    assert out == "" and not output.exists()  # фит не запускался


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--fix", "phi"], "имя=значение"),
        (["--fix", "gamma=1"], "gamma"),
        (["--bounds", "theta=100"], "имя=нижняя:верхняя"),
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
    code, _, err = run_cli(capsys, "fit")  # нет обязательного аргумента: у click это код 2, у нас 2 — флаги фита
    assert code == 1
    assert err


# --- panda profile ---


def write_md_files(directory: Path) -> tuple[Path, Path]:
    """Синтетическая траектория щели в формате GROMACS (.gro + .xtc): жидкость LIQ, стенки WAL."""
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
    names[liquid] = "O"  # известные элементы: MDAnalysis угадывает массы без предупреждений
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
    assert "2 блок" in out


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
    code, _, err = run_cli(
        capsys, "profile", str(topology), str(coordinates), "--pore-file", str(pore_file), "--select", "resname LIQ",
        "-o", str(tmp_path / "p.npz"),
    )  # fmt: skip
    assert code == 1
    assert "стен" in err


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
    assert "width_nm" in out  # размеры, найденные по траектории, показаны пользователю

    fit_output = tmp_path / "result.json"
    code, _, err = run_cli(
        capsys, "fit", str(output), "--pore-file", str(pore_output), "--shape", "roll", "--rho-bulk", "5",
        "-o", str(fit_output),
    )  # fmt: skip
    assert code in (0, 2), err
    assert json.loads(fit_output.read_text())["pore_name"] == "slit"


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
    assert read_values(tmp_path / "result.json") == pytest.approx(expected.values, abs=1e-9)  # путь — от конфига
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


# --- точка входа ---


def test_console_entry_point_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "panda_md", "shapes", "--pore", "slit"], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert "roll" in completed.stdout
