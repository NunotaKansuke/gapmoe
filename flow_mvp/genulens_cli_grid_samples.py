from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from genulens_cli_samples import (
    FEATURE_NAMES,
    RAW_COLUMNS,
    parse_verbosity3_stdout,
    physical_and_source_group_from_raw,
    remove_event_rate_weight,
    resample_weighted_indices,
    run_genulens_cli,
    ZeroSourceDensityError,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run genulens CLI for several (l,b) sightlines and build one conditional-flow sample table."
    )
    parser.add_argument("--genulens", type=Path, default=Path("../genulens/genulens"))
    parser.add_argument("--workdir", type=Path, default=Path("../genulens"))
    parser.add_argument(
        "--los",
        action="append",
        required=True,
        help="Sightline as 'l,b'. Repeat this option for multiple conditions.",
    )
    parser.add_argument("--n-simu-per-los", type=int, default=20_000)
    parser.add_argument("--resample-size-per-los", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remnant", type=int, choices=(0, 1), default=0)
    parser.add_argument("--binary", type=int, choices=(0, 1), default=0)
    parser.add_argument("--nsd", type=int, choices=(0, 1), default=0)
    parser.add_argument("--small-gamma", type=int, choices=(0, 1), default=0)
    parser.add_argument("--weight-mode", choices=("event", "base"), default="event")
    parser.add_argument("--source-group", type=int, choices=(3, 4), default=None)
    parser.add_argument("--skip-zero-density", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stdout-dir", type=Path, default=None)
    args = parser.parse_args()

    physical_parts = []
    condition_parts = []
    source_group_parts = []
    raw_physical_parts = []
    raw_weight_parts = []
    metadata = []
    resample_size = args.resample_size_per_los or args.n_simu_per_los

    tasks = [
        (
            idx,
            l_deg,
            b_deg,
            args.seed + idx,
            args.genulens,
            args.workdir,
            args.n_simu_per_los,
            resample_size,
            args.stdout_dir,
            args.remnant,
            args.binary,
            args.nsd,
            args.small_gamma,
            args.weight_mode,
            args.source_group,
            args.skip_zero_density,
        )
        for idx, (l_deg, b_deg) in enumerate(_parse_los(args.los))
    ]

    if args.stdout_dir is not None:
        args.stdout_dir.mkdir(parents=True, exist_ok=True)

    if args.jobs == 1:
        results = [_run_one(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_run_one, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

    results = [result for result in results if result is not None]
    if not results:
        raise ValueError("no sightlines produced source-group flow samples")

    for result in sorted(results, key=lambda item: item["idx"]):
        physical = result["physical"]
        condition = result["condition"]
        source_group = result["source_group"]
        raw_physical = result["raw_physical"]
        weights = result["weights"]
        physical_parts.append(physical)
        condition_parts.append(condition)
        source_group_parts.append(source_group)
        raw_physical_parts.append(raw_physical)
        raw_weight_parts.append(weights)
        metadata.append(result["metadata"])

    physical_all = np.concatenate(physical_parts, axis=0)
    condition_all = np.concatenate(condition_parts, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        physical=physical_all,
        condition=condition_all,
        source_group=np.concatenate(source_group_parts, axis=0),
        raw_physical=np.concatenate(raw_physical_parts, axis=0),
        raw_weights=np.concatenate(raw_weight_parts, axis=0),
        raw_columns=np.asarray(RAW_COLUMNS),
        feature_names=np.asarray(FEATURE_NAMES),
        condition_names=np.asarray(["l", "b"]),
        metadata=json.dumps(
            {
                "source": "genulens CLI stdout grid",
                "n_simu_per_los": int(args.n_simu_per_los),
                "resample_size_per_los": int(resample_size),
                "remnant": int(args.remnant),
                "binary": int(args.binary),
                "nsd": int(args.nsd),
                "small_gamma": int(args.small_gamma),
                "weight_mode": args.weight_mode,
                "source_group": args.source_group,
                "sightlines": metadata,
            },
            indent=2,
        ),
    )
    print(f"wrote {physical_all.shape[0]} conditional samples to {args.out}")
    print("x columns:", ", ".join(FEATURE_NAMES))
    print("condition columns: l, b")


def _parse_los(values: list[str]) -> list[tuple[float, float]]:
    out = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError(f"invalid --los value {value!r}; expected 'l,b'.")
        out.append((float(parts[0]), float(parts[1])))
    return out


def _run_one(task):
    (
        idx,
        l_deg,
        b_deg,
        seed,
        executable,
        workdir,
        n_simu,
        resample_size,
        stdout_dir,
        remnant,
        binary,
        nsd,
        small_gamma,
        weight_mode,
        source_group,
        skip_zero_density,
    ) = task
    print(f"running genulens l={l_deg:g} b={b_deg:g} Nsimu={n_simu} seed={seed}", flush=True)
    try:
        stdout = run_genulens_cli(
            executable=executable,
            workdir=workdir,
            l=l_deg,
            b=b_deg,
            n_simu=n_simu,
            seed=seed,
            remnant=remnant,
            binary=binary,
            nsd=nsd,
            small_gamma=small_gamma,
            source_group=source_group,
        )
    except ZeroSourceDensityError:
        if not skip_zero_density:
            raise
        print(f"skipping zero-density source group at l={l_deg:g} b={b_deg:g}", flush=True)
        return None
    if stdout_dir is not None:
        Path(stdout_dir).mkdir(parents=True, exist_ok=True)
        (Path(stdout_dir) / f"genulens_l{l_deg:g}_b{b_deg:g}.txt").write_text(stdout)

    raw = parse_verbosity3_stdout(stdout)
    raw_physical, weights, raw_source_group = physical_and_source_group_from_raw(raw)
    if weight_mode == "base":
        if small_gamma != 1:
            raise ValueError("weight_mode=base requires --small-gamma 1")
        weights = remove_event_rate_weight(raw_physical, weights)
    indices = resample_weighted_indices(
        raw_physical,
        weights,
        resample_size,
        seed=seed + 1009,
    )
    physical = raw_physical[indices]
    condition = np.broadcast_to(np.asarray([l_deg, b_deg], dtype=float), (physical.shape[0], 2))
    return {
        "idx": idx,
        "physical": physical,
        "condition": condition,
        "source_group": raw_source_group[indices],
        "raw_physical": raw_physical,
        "weights": weights,
        "metadata": {
            "l_deg": float(l_deg),
            "b_deg": float(b_deg),
            "n_raw": int(raw.shape[0]),
            "n_valid": int(raw_physical.shape[0]),
            "n_resampled": int(physical.shape[0]),
            "seed": int(seed),
            "remnant": int(remnant),
            "binary": int(binary),
            "nsd": int(nsd),
            "small_gamma": int(small_gamma),
            "weight_mode": weight_mode,
            "source_group": source_group,
        },
    }


if __name__ == "__main__":
    main()
