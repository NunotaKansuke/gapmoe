from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _run(task: tuple[str, str, float, float]) -> tuple[float, float, np.ndarray]:
    executable, workdir, l_deg, b_deg = task
    completed = subprocess.run(
        [
            executable,
            "l", str(l_deg),
            "b", str(b_deg),
            "SOURCE", "0",
            "NSD", "1",
            "REMNANT", "0",
            "BINARY", "0",
        ],
        cwd=workdir,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    rows = [
        [float(value) for value in line.split()]
        for line in completed.stdout.splitlines()
        if line and not line.startswith("#")
    ]
    return l_deg, b_deg, np.asarray(rows, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genulens-root", type=Path, default=Path("../genulens"))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.genulens_root.resolve()
    executable = root / "pre_gapmoe" / "calc_rho_profile"
    l_grid = np.arange(-5.0, 5.0 + 0.25, 0.5)
    b_grid = np.arange(-6.0, -2.0 + 0.25, 0.5)
    tasks = [(str(executable), str(root), float(l), float(b)) for b in b_grid for l in l_grid]
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(_run, tasks))

    first = results[0][2]
    distance_pc = first[:, 0]
    effective_source = np.empty((len(b_grid), len(l_grid), len(distance_pc), 11), dtype=float)
    for l_deg, b_deg, rows in results:
        if not np.array_equal(rows[:, 0], distance_pc):
            raise ValueError("calc_rho_profile distance grids differ between sightlines")
        il = int(round((l_deg - l_grid[0]) / 0.5))
        ib = int(round((b_deg - b_grid[0]) / 0.5))
        # Match genulens's unselected source and lens proposals exactly.
        # With no active luminosity-function cut, LineOfSightDensityGrid uses
        # rhoD_S = nMS * sqrt(DS / 8000) * 1e-3 (the default gammaDs=0.5),
        # while lenses are drawn from total number density rho, not rho*DL^2.
        # The released kernel is conditional on (DS, source group) after that
        # proposal and has its DL^2 thetaE mu_rel rate factor removed, so its
        # companion measure must be source proposal times the lens column.
        source_weight = np.sqrt(distance_pc / 8000.0) * 1.0e-3
        source = rows[:, 1:12] * source_weight[:, None]
        lens_integrand = rows[:, 24]
        lens_column = np.zeros_like(distance_pc)
        lens_column[1:] = np.cumsum(
            0.5 * (lens_integrand[1:] + lens_integrand[:-1]) * np.diff(distance_pc)
        )
        effective_source[ib, il] = source * lens_column[:, None]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        l_deg=l_grid,
        b_deg=b_grid,
        distance_pc=distance_pc,
        source_by_component=effective_source,
    )
    print(f"wrote {effective_source.shape} effective source grid to {args.out}")


if __name__ == "__main__":
    main()
