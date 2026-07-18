from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine residual-flow tables with balanced source-group sampling."
    )
    parser.add_argument("--samples", type=Path, action="append", required=True)
    parser.add_argument("--per-group", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rest_parts = []
    condition_parts = []
    for path in args.samples:
        data = np.load(path, allow_pickle=False)
        rest_parts.append(np.asarray(data["rest"], dtype=np.float32))
        condition_parts.append(np.asarray(data["condition"], dtype=np.float32))
    rest = np.concatenate(rest_parts)
    condition = np.concatenate(condition_parts)
    groups = np.argmax(condition[:, 3:], axis=1)

    rng = np.random.default_rng(args.seed)
    chosen_parts = []
    available = {}
    for group in range(5):
        indices = np.flatnonzero(groups == group)
        available[str(group)] = int(len(indices))
        if len(indices) == 0:
            raise ValueError(f"source group {group} has no training samples")
        chosen_parts.append(
            rng.choice(indices, size=args.per_group, replace=len(indices) < args.per_group)
        )
    chosen = np.concatenate(chosen_parts)
    rng.shuffle(chosen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        rest=rest[chosen],
        condition=condition[chosen],
        feature_names=np.asarray(["ML", "DL", "mu_E", "mu_N"]),
        condition_names=np.asarray([
            "l", "b", "DS", "source_group_thin", "source_group_thick",
            "source_group_bulge", "source_group_NSD", "source_group_halo",
        ]),
        metadata=json.dumps({
            "sources": [str(path) for path in args.samples],
            "available_by_group": available,
            "selected_per_group": int(args.per_group),
            "seed": int(args.seed),
        }, indent=2),
    )
    print(f"wrote {len(chosen)} balanced residual-flow samples to {args.out}")
    print("available source-group counts:", available)


if __name__ == "__main__":
    main()
