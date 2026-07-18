from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare source-group conditional residual-flow samples.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.samples, allow_pickle=False)
    physical = np.asarray(data["physical"], dtype=float)
    lb = np.asarray(data["condition"], dtype=float)
    groups = np.asarray(data["source_group"], dtype=int)
    if physical.shape[0] != lb.shape[0] or groups.shape != (physical.shape[0],):
        raise ValueError("physical, condition, and source_group rows must align")
    if np.any((groups < 0) | (groups >= 5)):
        raise ValueError("source_group must be in [0, 4]")
    rest = physical[:, [0, 1, 3, 4]]
    condition = np.concatenate([lb, physical[:, 2:3], np.eye(5)[groups]], axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        rest=rest,
        condition=condition,
        feature_names=np.asarray(["ML", "DL", "mu_E", "mu_N"]),
        condition_names=np.asarray(["l", "b", "DS", "source_group_thin", "source_group_thick", "source_group_bulge", "source_group_NSD", "source_group_halo"]),
        metadata=json.dumps({"source": str(args.samples), "num_samples": int(len(rest))}, indent=2),
    )
    print(f"wrote {len(rest)} residual-flow samples to {args.out}")


if __name__ == "__main__":
    main()
