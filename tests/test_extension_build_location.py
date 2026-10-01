"""Guard for #15: a rebuild that silently does not take effect.

`maturin develop` run from crates/proxide_py ignores the root pyproject's
[tool.maturin] table, so it installs the crate's cdylib as a TOP-LEVEL
`_proxider` module. `import proxide` loads `proxide._proxider` instead, and
keeps running whatever the last root-level build left in src/proxide/. Nothing
errors. These tests make that state fail loudly.
"""

import importlib.machinery
import importlib.util
from pathlib import Path

import proxide


def decoy_extension_locations(search_path: list[str]) -> list[str]:
    """Return the origin of every top-level `_proxider` findable on `search_path`."""
    # PathFinder caches directory listings; without this, a decoy written after
    # a directory was first scanned is invisible (caught by the negative control).
    importlib.invalidate_caches()
    found = []
    for entry in search_path:
        spec = importlib.machinery.PathFinder.find_spec("_proxider", [entry])
        if spec is not None:
            found.append(str(spec.origin or spec.submodule_search_locations))
    return found


def test_no_top_level_proxider_decoy_is_importable():
    import sys

    decoys = decoy_extension_locations(sys.path)
    assert not decoys, (
        "A top-level `_proxider` module is importable, so a Rust rebuild went to a "
        "location `import proxide` never loads (#15). Delete it and rebuild from "
        f"the repo root (`maturin develop --release`). Found: {decoys}"
    )


def test_proxide_extension_is_loaded_from_inside_the_package():
    spec = importlib.util.find_spec("proxide._proxider")
    assert spec is not None and spec.origin is not None, "proxide._proxider is not built"
    package_dir = Path(proxide.__file__).resolve().parent
    assert Path(spec.origin).resolve().parent == package_dir, (
        f"proxide._proxider loads from {spec.origin}, not from the package at {package_dir}"
    )


def test_decoy_detector_fires_on_a_bare_extension_file(tmp_path):
    """Negative control: the shape a crate-dir build can leave on sys.path."""
    assert decoy_extension_locations([str(tmp_path)]) == []
    (tmp_path / "_proxider.abi3.so").write_bytes(b"")
    assert decoy_extension_locations([str(tmp_path)]) == [str(tmp_path / "_proxider.abi3.so")]


def test_decoy_detector_fires_on_a_package_directory(tmp_path):
    """Negative control: maturin's `_proxider/` package-directory install shape."""
    pkg = tmp_path / "_proxider"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    assert decoy_extension_locations([str(tmp_path)]) == [str(pkg / "__init__.py")]
