from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from gapmoe.pre_runner import PreRunner
from gapmoe.source_selection import CmdCoordinates, CmdPriorTable, GenulensSourceModel, SourceEvidenceGrid, SourceSelection


def test_pre_runner_environment_reports_missing_tools(tmp_path: Path) -> None:
    genulens_root = tmp_path / "genulens"
    pre_gapmoe = genulens_root / "pre_gapmoe"
    pre_gapmoe.mkdir(parents=True)
    (pre_gapmoe / "calc_mass_dist").write_text("")

    runner = PreRunner(genulens_root=genulens_root, output_dir=tmp_path / "out")
    env = runner.check_environment()

    assert env.genulens_root == genulens_root.resolve()
    assert env.pre_gapmoe_dir == pre_gapmoe.resolve()
    assert env.available_tools == ("calc_mass_dist",)
    assert env.missing_tools == ("calc_rho_profile", "calc_murel_dist")
    assert not env.ok


def test_pre_runner_can_use_genulens_python_api(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class Table:
        stdout = "# Columns:\n# x  y\n1  2\n"

    def make_function(name):
        def run(**kwargs):
            calls.append((name, kwargs))
            return Table()

        return run

    fake_genulens = SimpleNamespace(
        __file__=str(tmp_path / "genulens.so"),
        pre_gapmoe=SimpleNamespace(
            mass_distribution=make_function("mass"),
            rho_profile=make_function("rho"),
            murel_distribution=make_function("murel"),
        ),
    )
    monkeypatch.setitem(sys.modules, "genulens", fake_genulens)

    runner = PreRunner(output_dir=tmp_path / "out", backend="auto")
    assert runner.backend == "python"
    env = runner.check_environment()
    assert env.backend == "python"
    assert env.ok

    cmd_prior = CmdPriorTable(
        coordinates=CmdCoordinates(reference_band="Imag", blue_band="Vmag", red_band="Imag"),
        reference_edges=np.asarray([0.0, 1.0]),
        color_edges=np.asarray([0.0, 1.0]),
        density_by_component=np.ones((11, 1, 1)),
    )
    result = runner.run(
        l=1.0,
        b=-3.9,
        run_name="api",
        distance_max_pc=1000,
        rho_step_pc=500,
        murel_distance_step_pc=500,
        n_simu=100,
        source_model=GenulensSourceModel(source_data=SourceSelection()),
        cmd_prior=cmd_prior,
    )

    assert result.mass_path.read_text() == Table.stdout
    assert result.rho_path.read_text() == Table.stdout
    assert result.murel_path.read_text() == Table.stdout
    assert [name for name, _ in calls] == ["mass", "rho", "murel"]
    assert calls[1][1]["SOURCE"] == 0
    assert calls[2][1]["GRID"] == 1
    assert calls[2][1]["SOURCEGROUPS"] == 1
    assert result.source_evidence_path is not None
    assert SourceEvidenceGrid.load_npz(result.source_evidence_path).evidence_by_component.shape == (1, 11)
    assert result.cmd_prior_path is not None
    assert CmdPriorTable.load_npz(result.cmd_prior_path).coordinates == cmd_prior.coordinates
