from __future__ import annotations

from pathlib import Path

from gapmoe import PreRunner


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
