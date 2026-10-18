"""Version of the installed library: written into results and saved files (reproducibility of the computation)."""

from importlib.metadata import PackageNotFoundError, version

DISTRIBUTION_NAME = "panda-md"
UNKNOWN_VERSION = "0.0.0"  # package is not installed (run from sources without installation)


def library_version() -> str:
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return UNKNOWN_VERSION
