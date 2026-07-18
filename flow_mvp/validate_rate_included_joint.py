"""Compare a rate-included source grid and kernel with independent genulens events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from gapmoe.density.flow_backend import EventKernelFlow


SOURCE_GROUP_BY_COMPONENT = np.asarray((0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4))
KAPPA_MAS_PER_MSUN = 8.144
PHYSICAL_NAMES = ("ML", "DL", "DS", "mu_E", "mu_N")


def ks(x: np.ndarray, y: np.ndarray) -> float:
    values = np.sort(np.concatenate((x, y)))
    fx = np.searchsorted(np.sort(x), values, side="right") / len(x)
    fy = np.searchsorted(np.sort(y), values, side="right") / len(y)
    return float(np.max(np.abs(fx - fy)))


def derived_observables(physical: np.ndarray) -> dict[str, np.ndarray]:
    """Return common microlensing observables implied by the 5D state."""

    ml, dl, ds, mu_e, mu_n = np.asarray(physical, dtype=float).T
    mu = np.hypot(mu_e, mu_n)
    pi_rel = 1.0 / dl - 1.0 / ds
    theta_e = np.sqrt(KAPPA_MAS_PER_MSUN * ml * pi_rel)
    return {
        "DL_over_DS": dl / ds,
        "mu_rel": mu,
        "theta_E": theta_e,
        "t_E_days": theta_e / mu * 365.25,
    }


def rank_columns(values: np.ndarray) -> np.ndarray:
    """Average ranks by column, including correct handling of tied grid DS."""

    values = np.asarray(values, dtype=float)
    ranked = np.empty_like(values)
    for column in range(values.shape[1]):
        _, inverse, counts = np.unique(
            values[:, column], return_inverse=True, return_counts=True
        )
        starts = np.cumsum(counts) - counts
        average_ranks = starts + 0.5 * (counts - 1)
        ranked[:, column] = average_ranks[inverse]
    return ranked


def max_rank_correlation_difference(x: np.ndarray, y: np.ndarray) -> float:
    """Maximum Spearman correlation error; robust to rare extreme tails."""

    difference = (
        np.corrcoef(rank_columns(x), rowvar=False)
        - np.corrcoef(rank_columns(y), rowvar=False)
    )
    return float(np.max(np.abs(difference)))


def quantile_error_iqr(x: np.ndarray, y: np.ndarray) -> float:
    """Maximum 1/5/50/95/99-percentile error in target-IQR units."""

    probabilities = (0.01, 0.05, 0.5, 0.95, 0.99)
    scale = max(float(np.subtract(*np.quantile(x, (0.75, 0.25)))), 1.0e-12)
    difference = np.abs(np.quantile(y, probabilities) - np.quantile(x, probabilities))
    return float(np.max(difference) / scale)


def interpolate_grid(data, l_value: float, b_value: float) -> np.ndarray:
    l_grid, b_grid = data["l_deg"], data["b_deg"]
    values = data["source_by_component"]
    il1 = min(np.searchsorted(l_grid, l_value, side="right"), len(l_grid) - 1)
    ib1 = min(np.searchsorted(b_grid, b_value, side="right"), len(b_grid) - 1)
    il0, ib0 = max(0, il1 - 1), max(0, ib1 - 1)
    tl = 0.0 if il0 == il1 else (l_value - l_grid[il0]) / (l_grid[il1] - l_grid[il0])
    tb = 0.0 if ib0 == ib1 else (b_value - b_grid[ib0]) / (b_grid[ib1] - b_grid[ib0])
    return (
        (1 - tb) * ((1 - tl) * values[ib0, il0] + tl * values[ib0, il1])
        + tb * ((1 - tl) * values[ib1, il0] + tl * values[ib1, il1])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-marginal-ks", type=float, default=0.05)
    parser.add_argument("--max-derived-ks", type=float, default=0.05)
    parser.add_argument("--max-rank-correlation-difference", type=float, default=0.10)
    parser.add_argument("--max-group-difference", type=float, default=0.02)
    args = parser.parse_args()

    grid = np.load(args.grid, allow_pickle=False)
    target = np.load(args.holdout, allow_pickle=False)
    model = EventKernelFlow.load(args.kernel)
    rng = np.random.default_rng(args.seed)
    reports = []
    for cell_index, (l_value, b_value) in enumerate(np.unique(target["condition"], axis=0)):
        selected = np.all(target["condition"] == (l_value, b_value), axis=1)
        target_physical = np.asarray(target["physical"][selected], dtype=float)
        target_group = np.asarray(target["source_group"][selected], dtype=int)
        n = len(target_physical)

        source = interpolate_grid(grid, float(l_value), float(b_value))
        distance = np.asarray(grid["distance_pc"], dtype=float) / 1000.0
        widths = np.empty_like(distance)
        widths[0] = 0.5 * (distance[1] - distance[0])
        widths[-1] = 0.5 * (distance[-1] - distance[-2])
        widths[1:-1] = 0.5 * (distance[2:] - distance[:-2])
        probability = (np.maximum(source, 0.0) * widths[:, None]).ravel()
        probability /= probability.sum()
        flat = rng.choice(len(probability), size=n, p=probability)
        ds = distance[flat // 11]
        component = flat % 11
        group = SOURCE_GROUP_BY_COMPONENT[component]

        condition = np.zeros((n, 8), dtype=np.float32)
        condition[:, 0], condition[:, 1], condition[:, 2] = l_value, b_value, ds
        condition[np.arange(n), 3 + group] = 1.0
        keys = jax.random.split(jax.random.key(args.seed + cell_index), n)
        generated_rest = np.asarray(jax.block_until_ready(
            jax.jit(jax.vmap(model.sample))(keys, condition)
        ), dtype=float)
        generated = np.column_stack((
            generated_rest[:, 0], generated_rest[:, 1], ds,
            generated_rest[:, 2], generated_rest[:, 3],
        ))
        group_target = np.bincount(target_group, minlength=5) / n
        group_generated = np.bincount(group, minlength=5) / n
        target_derived = derived_observables(target_physical)
        generated_derived = derived_observables(generated)
        reports.append({
            "l_deg": float(l_value),
            "b_deg": float(b_value),
            "n": n,
            "ks": {
                name: ks(target_physical[:, index], generated[:, index])
                for index, name in enumerate(PHYSICAL_NAMES)
            },
            "derived_ks": {
                name: ks(target_derived[name], generated_derived[name])
                for name in target_derived
            },
            "quantile_error_iqr": {
                name: quantile_error_iqr(target_physical[:, index], generated[:, index])
                for index, name in enumerate(PHYSICAL_NAMES)
            },
            "max_rank_correlation_difference": max_rank_correlation_difference(
                target_physical,
                generated,
            ),
            "group_target": group_target.tolist(),
            "group_generated": group_generated.tolist(),
            "group_max_abs_difference": float(np.max(np.abs(group_target - group_generated))),
        })
        print(f"validated joint cell {cell_index + 1}", flush=True)
    maxima = {
        "marginal_ks": max(value for report in reports for value in report["ks"].values()),
        "derived_ks": max(value for report in reports for value in report["derived_ks"].values()),
        "rank_correlation_difference": max(
            report["max_rank_correlation_difference"] for report in reports
        ),
        "source_group_difference": max(report["group_max_abs_difference"] for report in reports),
    }
    thresholds = {
        "marginal_ks": args.max_marginal_ks,
        "derived_ks": args.max_derived_ks,
        "rank_correlation_difference": args.max_rank_correlation_difference,
        "source_group_difference": args.max_group_difference,
    }
    passed = all(maxima[name] <= thresholds[name] for name in thresholds)
    payload = {
        "acceptance": {"passed": passed, "maxima": maxima, "thresholds": thresholds},
        "reports": reports,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(reports)} joint reports to {args.out}")
    print(f"acceptance: {'PASS' if passed else 'FAIL'}; maxima={maxima}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
