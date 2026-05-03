from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_code(path: Path) -> tuple[list, str]:
    notebook = json.loads(path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    return code_cells, source


def _assert_clean(code_cells) -> None:
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)


def _assert_compiles(path: Path, code_cells) -> None:
    for index, cell in enumerate(code_cells):
        compile("".join(cell.get("source", [])), f"{path}:code-cell-{index}", "exec")


def test_pre_runner_notebook_code_compiles_and_is_clear() -> None:
    path = ROOT / "example" / "pre_runner.ipynb"
    code_cells, source = _load_code(path)
    _assert_clean(code_cells)

    assert "PreRunner" in source
    assert "runner.run(" in source
    assert "HistogramDensity.from_paths(" in source
    assert "GalacticPrior(density)" in source
    assert "prior.log_prob(" in source

    _assert_compiles(path, code_cells)


def test_emcee_physical_params_notebook_code_compiles_and_is_clear() -> None:
    path = ROOT / "example" / "emcee_physical_params.ipynb"
    code_cells, source = _load_code(path)
    _assert_clean(code_cells)

    assert "HistogramDensity.from_paths(" in source
    assert "GalacticPrior(density)" in source
    assert "prior.log_prob(*theta)" in source
    assert "corner.corner(chain, labels=labels)" in source
    assert "PreRunner" not in source

    _assert_compiles(path, code_cells)


def test_emcee_binary_circular_notebook_code_compiles_and_is_clear() -> None:
    path = ROOT / "example" / "emcee_binary_circular.ipynb"
    code_cells, source = _load_code(path)
    _assert_clean(code_cells)

    assert "HistogramDensity.from_paths(" in source
    assert "BinaryCircularParameterization" in source
    assert "calc_vEarth" in source
    assert "param.to_physical" in source
    assert "param.log_abs_det_jacobian" in source
    assert "GalacticPrior(density, parameterization=param)" in source
    assert "prior.log_prob(theta" in source
    assert "corner.corner(phys_samples" in source
    assert "PreRunner" not in source

    _assert_compiles(path, code_cells)


def test_jax_prior_notebook_code_compiles_and_is_clear() -> None:
    path = ROOT / "example" / "jax_prior.ipynb"
    code_cells, source = _load_code(path)
    _assert_clean(code_cells)

    assert "HistogramDensity.from_paths(" in source
    assert "JaxHistogramDensity.from_numpy" in source
    assert "JaxGalacticPrior" in source
    assert "jax.jit" in source
    assert "jax.vmap" in source
    assert "jax.grad" in source
    assert "BinaryCircularParameterization" in source
    assert "calc_vEarth" in source
    assert "PreRunner" not in source

    _assert_compiles(path, code_cells)
