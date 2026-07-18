"""Build p_event(DS, source component | l, b) from raw genulens wtj tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from genulens_cli_samples import parse_verbosity3_stdout, valid_raw_event_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--stdout-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step-pc", type=float, default=100.0)
    parser.add_argument("--max-distance-pc", type=float, default=16_000.0)
    args = parser.parse_args()

    with np.load(args.samples, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
    if metadata.get("weight_mode") != "event":
        raise ValueError("rate-included source grid requires weight_mode=event")
    if metadata.get("small_gamma") != 1 or metadata.get("nsd") != 1:
        raise ValueError("rate-included source grid requires SMALLGAMMA=1 and NSD=1")
    if metadata.get("remnant") != 0 or metadata.get("binary") != 0:
        raise ValueError("rate-included-v1 source grid requires REMNANT=0 and BINARY=0")

    sightlines = metadata["sightlines"]
    l_deg = np.asarray(sorted({float(row["l_deg"]) for row in sightlines}))
    b_deg = np.asarray(sorted({float(row["b_deg"]) for row in sightlines}))
    distance_pc = np.arange(args.step_pc, args.max_distance_pc + 0.5 * args.step_pc, args.step_pc)
    edges = np.concatenate(
        ([distance_pc[0] - 0.5 * args.step_pc], distance_pc + 0.5 * args.step_pc)
    )
    source = np.zeros((len(b_deg), len(l_deg), len(distance_pc), 11), dtype=np.float64)

    for index, row in enumerate(sightlines, start=1):
        l_value, b_value = float(row["l_deg"]), float(row["b_deg"])
        path = args.stdout_dir / f"genulens_l{l_value:g}_b{b_value:g}.txt"
        raw = parse_verbosity3_stdout(path.read_text())
        valid = valid_raw_event_mask(raw)
        raw = raw[valid]
        weights, ds, component = raw[:, 0], raw[:, 3], raw[:, 14].astype(int)
        ib = int(np.searchsorted(b_deg, b_value))
        il = int(np.searchsorted(l_deg, l_value))
        for source_component in range(11):
            chosen = component == source_component
            if np.any(chosen):
                counts, _ = np.histogram(ds[chosen], bins=edges, weights=weights[chosen])
                # Stored values are a density per kpc, matching the trapezoid
                # integration convention used by FlowSourceDistanceGrid.
                source[ib, il, :, source_component] = counts / (args.step_pc / 1000.0)
        if index % 20 == 0 or index == len(sightlines):
            print(f"processed {index}/{len(sightlines)} sightlines", flush=True)

    if not np.isfinite(source).all() or np.any(source < 0.0):
        raise ValueError("invalid rate-included source grid")
    if np.any(np.sum(source, axis=(2, 3)) <= 0.0):
        raise ValueError("at least one sightline has zero total event measure")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        l_deg=l_deg,
        b_deg=b_deg,
        distance_pc=distance_pc,
        source_by_component=source,
    )
    print(f"wrote rate-included source grid {source.shape} to {args.out}")


if __name__ == "__main__":
    main()
