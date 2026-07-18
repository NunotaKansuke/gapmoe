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


def _assert_executed(code_cells) -> None:
    assert code_cells
    assert all(isinstance(cell.get("execution_count"), int) for cell in code_cells)
    assert all(
        output.get("output_type") != "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )


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
    assert "density.log_density(" in source

    _assert_compiles(path, code_cells)


def test_emcee_physical_params_notebook_code_compiles_and_is_clear() -> None:
    path = ROOT / "example" / "emcee_physical_params.ipynb"
    code_cells, source = _load_code(path)
    _assert_executed(code_cells)

<<<<<<< HEAD
    assert "HistogramDensity.from_paths(" in source
    assert "GalacticModel(density)" in source
    assert "prior.log_prob(*theta)" in source
=======
    assert "Model().resume(" in source
    assert "model.isochrone(" in source
    assert "model.galactic_model(isochrone)" in source
    assert "compiled_log_density" in source
>>>>>>> codex/inference-mode-cleanup
    assert "genulens_out.dat" in source
    assert "genulens_weights" in source
    assert "fig = corner.corner(" in source
    assert "    chain," in source
    assert "corner.corner(\n    genulens_chain" in source
    assert "PreRunner" not in source

    _assert_compiles(path, code_cells)


def test_flow_galactic_model_notebook_is_executed_and_complete() -> None:
    path = ROOT / "example" / "flow_galactic_model.ipynb"
    code_cells, source = _load_code(path)
    _assert_executed(code_cells)

    assert '.set_flow(release="rate-included-v1")' in source
    assert "prior.log_density(theta)" in source
    assert "emcee.EnsembleSampler(" in source
    assert "genulens_flow_out.dat" in source
    assert "weights = rows[:, 0]" in source
    assert "weighted median comparison" in source
    assert "magnitude_range=(15.0, 21.0)" in source
    assert "magnitudes=source_magnitudes" in source
    assert "log_source_density(" in source
    assert "source_radius(" in source

    _assert_compiles(path, code_cells)


def test_flow_genulens_snapshot_matches_release_options() -> None:
    first_line = (ROOT / "example" / "genulens_flow_out.dat").read_text().splitlines()[0]

    assert "REMNANT 0" in first_line
    assert "BINARY 0" in first_line
    assert "NSD 1" in first_line
    assert "SMALLGAMMA 1" in first_line
