"""Layer rules (specification §2): the core knows nothing of pores, trajectories, analysis and interfaces; heavy
optional packages are imported only where they are needed."""

import subprocess
import sys

import pytest

# module → packages that must not be in sys.modules after importing it
FORBIDDEN = {
    "panda_md.core.fitting": (
        "panda_md.pores",
        "panda_md.trajectory",
        "panda_md.analysis",
        "panda_md.synthetic",
        "panda_md.plotting",
        "panda_md.cli",
    ),
    "panda_md.core.model": ("panda_md.pores", "panda_md.trajectory", "panda_md.analysis", "panda_md.cli"),
    "panda_md.pores.registry": ("panda_md.trajectory", "panda_md.analysis", "panda_md.plotting", "panda_md.cli"),
    "panda_md": ("MDAnalysis", "matplotlib", "typer"),
    "panda_md.cli": ("MDAnalysis", "matplotlib"),
}


@pytest.mark.parametrize(("module", "forbidden"), sorted(FORBIDDEN.items()))
def test_module_does_not_pull_forbidden_packages(module: str, forbidden: tuple[str, ...]) -> None:
    # the panda_md package is replaced by an empty module: otherwise its __init__ imports everything and the
    # dependency graph is hidden
    stub = (
        "import importlib, importlib.util, sys, types; "
        "location = importlib.util.find_spec('panda_md').submodule_search_locations; "
        "package = types.ModuleType('panda_md'); package.__path__ = list(location); sys.modules['panda_md'] = package; "
    )
    load = "" if module == "panda_md" else stub
    code = (
        f"{load}import sys, importlib; importlib.import_module({module!r}); "
        f"print(','.join(sorted(name for name in sys.modules if name.split('.')[0] in {{'MDAnalysis', 'matplotlib', "
        f"'typer'}} or name.startswith('panda_md.'))))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    loaded = completed.stdout.strip().split(",")
    offenders = sorted(
        {name for name in loaded for prefix in forbidden if name == prefix or name.startswith(prefix + ".")}
    )
    assert offenders == [], f"{module} pulls in {offenders}"
