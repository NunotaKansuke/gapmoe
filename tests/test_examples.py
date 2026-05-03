from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_emcee_physical_params_notebook_code_compiles_and_is_clear() -> None:
    path = ROOT / "example" / "emcee_physical_params.ipynb"
    notebook = json.loads(path.read_text())

    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)

    source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    assert "HistogramDensity.from_pre_run(pre_run)" in source
    assert "GalacticPrior(density)" in source
    assert "prior.log_prob(*theta)" in source
    assert "corner.corner(chain, labels=labels)" in source
    assert "GalacticModel" not in source
    assert "PhysicalParams" not in source

    for index, cell in enumerate(code_cells):
        compile("".join(cell.get("source", [])), f"{path}:code-cell-{index}", "exec")


def test_emcee_binary_circular_notebook_code_compiles_and_is_clear() -> None:
    path = ROOT / "example" / "emcee_binary_circular.ipynb"
    notebook = json.loads(path.read_text())

    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)

    source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    assert "BinaryCircularParameterization" in source
    assert "calc_vEarth" in source
    assert "param.to_physical" in source
    assert "param.log_abs_det_jacobian" in source
    assert "GalacticPrior(density, parameterization=param)" in source
    assert "prior.log_prob(theta" in source
    assert "corner.corner(phys_samples" in source

    for index, cell in enumerate(code_cells):
        compile("".join(cell.get("source", [])), f"{path}:code-cell-{index}", "exec")


def test_jax_prior_notebook_code_compiles_and_is_clear() -> None:
    path = ROOT / "example" / "jax_prior.ipynb"
    notebook = json.loads(path.read_text())

    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)

    source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    assert "JaxHistogramDensity.from_numpy" in source
    assert "JaxGalacticPrior" in source
    assert "jax.jit" in source
    assert "jax.vmap" in source
    assert "jax.grad" in source
    assert "BinaryCircularParameterization" in source
    assert "calc_vEarth" in source

    for index, cell in enumerate(code_cells):
        compile("".join(cell.get("source", [])), f"{path}:code-cell-{index}", "exec")
