"""Validate a residual event-kernel flow across (l, b, source group) cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from gapmoe.density.flow_backend import EventKernelFlow
from validate_rate_included_joint import (
    derived_observables,
    max_rank_correlation_difference,
)


def _ks(x: np.ndarray, y: np.ndarray) -> float:
    values = np.sort(np.concatenate([x, y]))
    return float(np.max(np.abs(np.searchsorted(np.sort(x), values, side="right") / len(x) - np.searchsorted(np.sort(y), values, side="right") / len(y))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--max-per-cell", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-marginal-ks", type=float, default=0.05)
    parser.add_argument("--max-derived-ks", type=float, default=0.05)
    parser.add_argument("--max-rank-correlation-difference", type=float, default=0.10)
    parser.add_argument(
        "--source-groups",
        default="0,1,2,3,4",
        help="comma-separated source groups to validate",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_groups = tuple(int(value) for value in args.source_groups.split(","))
    if not source_groups or any(group < 0 or group > 4 for group in source_groups):
        parser.error("--source-groups must contain values in [0, 4]")

    data = np.load(args.samples, allow_pickle=False)
    rest = np.asarray(data["rest"], dtype=float)
    condition = np.asarray(data["condition"], dtype=float)
    model = EventKernelFlow.load(args.model_dir)
    rng = np.random.default_rng(args.seed)
    reports = []
    # `condition` can contain tens of millions of rows.  Iterating through it
    # in Python just to find the sightlines dominates the validation runtime.
    sightlines = np.unique(condition[:, :2], axis=0)
    sightlines = sightlines[np.lexsort((sightlines[:, 1], sightlines[:, 0]))]
    sample_many = jax.jit(jax.vmap(model.sample))
    log_density_many = jax.jit(model.log_density)
    for l_deg, b_deg in sightlines:
        for group in source_groups:
            mask = (condition[:, 0] == l_deg) & (condition[:, 1] == b_deg) & (np.argmax(condition[:, 3:], axis=1) == group)
            indices = np.flatnonzero(mask)
            if len(indices) == 0:
                continue
            chosen = rng.choice(indices, size=min(len(indices), args.max_per_cell), replace=False)
            target = rest[chosen]
            target_condition = condition[chosen]
            keys = jax.random.split(jax.random.key(args.seed + len(reports)), len(chosen))
            generated = np.asarray(sample_many(keys, target_condition))
            log_density = np.asarray(log_density_many(target, target_condition))
            target_physical = np.column_stack((
                target[:, 0], target[:, 1], target_condition[:, 2],
                target[:, 2], target[:, 3],
            ))
            generated_physical = np.column_stack((
                generated[:, 0], generated[:, 1], target_condition[:, 2],
                generated[:, 2], generated[:, 3],
            ))
            target_derived = derived_observables(target_physical)
            generated_derived = derived_observables(generated_physical)
            reports.append({
                "l_deg": l_deg,
                "b_deg": b_deg,
                "source_group": group,
                "n": int(len(chosen)),
                "mean_log_density": float(np.mean(log_density)),
                "ks": {name: _ks(target[:, index], generated[:, index]) for index, name in enumerate(("ML", "DL", "mu_E", "mu_N"))},
                "derived_ks": {
                    name: _ks(target_derived[name], generated_derived[name])
                    for name in target_derived
                },
                "max_rank_correlation_difference": max_rank_correlation_difference(
                    target_physical, generated_physical
                ),
            })
            if args.progress_every and len(reports) % args.progress_every == 0:
                print(f"validated {len(reports)} cells", flush=True)
    maxima = {
        "marginal_ks": max(value for report in reports for value in report["ks"].values()),
        "derived_ks": max(value for report in reports for value in report["derived_ks"].values()),
        "rank_correlation_difference": max(
            report["max_rank_correlation_difference"] for report in reports
        ),
    }
    thresholds = {
        "marginal_ks": args.max_marginal_ks,
        "derived_ks": args.max_derived_ks,
        "rank_correlation_difference": args.max_rank_correlation_difference,
    }
    passed = all(maxima[name] <= thresholds[name] for name in thresholds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model_dir": str(args.model_dir),
        "source_groups": source_groups,
        "acceptance": {"passed": passed, "maxima": maxima, "thresholds": thresholds},
        "reports": reports,
    }, indent=2) + "\n")
    print(f"wrote {len(reports)} validation cells to {args.out}")
    print(f"acceptance: {'PASS' if passed else 'FAIL'}; maxima={maxima}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
