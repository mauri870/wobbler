"""`wobbler.__version__` is the version of the installed package and agrees with pyproject.toml, so that a release has one place to change."""
import tomllib
from importlib.metadata import version
from pathlib import Path

import wobbler

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_version_is_the_one_of_the_package_metadata_and_of_pyproject():
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    assert wobbler.__version__ == version("wobbler") == declared


def test_the_version_is_a_dotted_number_and_is_not_a_lazily_loaded_name():
    assert all(part.isdigit() for part in wobbler.__version__.split(".")[:3])
    assert "__version__" in vars(wobbler), "the version must be bound at import, not through the lazy loader"


def test_a_source_tree_that_is_not_installed_still_imports_with_a_placeholder_version(monkeypatch):
    """The package is imported from src/ through PYTHONPATH by scripts that do not install it; the metadata is then missing and the import must not fail."""
    import importlib
    import importlib.metadata

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    try:
        importlib.reload(wobbler)
        assert wobbler.__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(wobbler)
    assert wobbler.__version__ == version("wobbler")
