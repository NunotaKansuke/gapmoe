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
    assert "gapmoe.Histogram.open(" in source

    _assert_compiles(path, code_cells)


def test_emcee_physical_params_notebook_code_compiles_and_is_clear() -> None:
    path = ROOT / "example" / "emcee_physical_params.ipynb"
    code_cells, source = _load_code(path)
    _assert_clean(code_cells)

    assert "gapmoe.Histogram.open(" in source
    assert "gapmoe.Model(" in source
    assert "physical_density.log_density(theta)" in source
    assert "genulens_out.dat" in source
    assert "genulens_weights" in source
    assert "fig = corner.corner(" in source
    assert "    chain," in source
    assert "corner.corner(\n    genulens_chain" in source
    assert "PreRunner" not in source

    _assert_compiles(path, code_cells)
