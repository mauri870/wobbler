"""The dependencies and the text of the package: no module imports anything but the standard library, numpy, torch (where a module has a torch path) and the optional packages named below, and no file of the package or the repository names the project that it was written for.
A structure check on the source."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "wobbler"
ALLOWED = {"numpy", "torch"}
OPTIONAL = {"scan.py": {"gaiamock"}, "sources.py": {"astroquery", "pyvo"}, "results.py": {"pandas"}}
NUMPY_ONLY = {"light.py", "sources.py", "results.py"}  # modules that must work where torch is not installed
HOST = "star" + "lord"  # the name of the project the package was written for, spelled in two parts so that this file does not hold it
INTERNAL = re.compile(rf"experiment\s+\d|\bW\d{{3}}\b|PITCH|{HOST}-\d|queue item|the owner|the manager|\bTIC[\s_-]?\d+|{HOST}", re.IGNORECASE)
DYNAMIC_IMPORTS = {"__import__", "import_module"}
DYNAMIC_ALLOWED = {"__init__.py"}  # the lazy loader of the package's own submodules


def imported(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield ("." * node.level) + (node.module or "")


def dynamic_imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if name in DYNAMIC_IMPORTS:
                yield name


def outside_the_allowed(path):
    names = {name for name in imported(path) if not name.startswith(".")}
    allowed = (set() if path.name in NUMPY_ONLY else ALLOWED) | {"numpy"} | OPTIONAL.get(path.name, set())
    return sorted({name.split(".")[0] for name in names} - allowed - set(sys.stdlib_module_names) - {"wobbler"})


def test_no_module_imports_anything_but_its_dependencies():
    files = sorted(SOURCE.rglob("*.py"))
    assert files, "the scan found no module of the package"
    found = {str(path.relative_to(SOURCE)): outside_the_allowed(path) for path in files}
    assert not any(found.values()), f"imports outside the standard library, numpy and torch: {found}"
    strings = {str(path.relative_to(SOURCE)): sorted(set(dynamic_imports(path))) for path in files if path.name not in DYNAMIC_ALLOWED}
    assert not any(strings.values()), f"imports by a string: {strings}"


def test_no_file_of_the_repository_names_the_project_it_was_written_for():
    hits = {}
    for path in sorted(ROOT.rglob("*")):
        if ".git" in path.parts or ".venv" in path.parts or "__pycache__" in path.parts or not path.is_file() or path.suffix not in (".py", ".md", ".toml", ".yml", ".txt", ".json", ".tex", ".cff", ".sh", ".csv"):
            continue
        if path.name == "test_boundary.py":
            continue
        found = INTERNAL.findall(path.read_text())
        if found:
            hits[str(path.relative_to(ROOT))] = found[:3]
    assert not hits, f"project-internal wording: {hits}"


def test_the_scan_sees_an_import_of_another_package_and_internal_wording(tmp_path):
    """The edge case that would hide a failure: a scan that finds nothing in a module that has something."""
    (tmp_path / "bad.py").write_text("import os\nimport pandas\n\ndef f():\n    from scipy import stats\n    import numpy\n")
    assert outside_the_allowed(tmp_path / "bad.py") == ["pandas", "scipy"]
    assert INTERNAL.search("as measured in experiment 1208") and INTERNAL.search("the owner asked") and INTERNAL.search(f"the host is {HOST} here") and INTERNAL.search("a star, TIC 123456789, is") and INTERNAL.search("TIC_123456")
    (tmp_path / "dynamic.py").write_text("import importlib\n\nmodule = importlib.import_module('x')\nother = __import__('pandas')\n")
    assert list(dynamic_imports(tmp_path / "dynamic.py")) == ["import_module", "__import__"]
    (tmp_path / "light.py").write_text("import torch\n")
    assert outside_the_allowed(tmp_path / "light.py") == ["torch"], "light.py must not import torch"
    (tmp_path / "scan.py").write_text("def f():\n    import gaiamock\n    import torch\n    import pandas\n")
    assert outside_the_allowed(tmp_path / "scan.py") == ["pandas"], "gaiamock is allowed in scan.py only"
