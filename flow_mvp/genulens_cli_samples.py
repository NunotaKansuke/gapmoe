from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


FEATURE_NAMES = ("ML", "DL", "DS", "mu_E", "mu_N")
RAW_COLUMNS = (
    "wtj",
    "M_L",
    "D_L_pc",
    "D_S_pc",
    "t_E",
    "theta_E",
    "pi_E",
    "pi_EN",
    "pi_EE",
    "mu_rel",
    "mu_Sl",
    "mu_Sb",
    "I_L",
    "K_L",
    "iS",
    "iL",
    "fREM",
)


class ZeroSourceDensityError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the genulens CLI, parse VERBOSITY=3 stdout, and resample weighted raw events to 5D flow samples."
    )
    parser.add_argument("--genulens", type=Path, default=Path("../genulens/genulens"))
    parser.add_argument("--workdir", type=Path, default=Path("../genulens"))
    parser.add_argument("--l", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=-3.9)
    parser.add_argument("--n-simu", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample-size", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stdout-out", type=Path, default=None)
    args = parser.parse_args()

    stdout = run_genulens_cli(
        executable=args.genulens,
        workdir=args.workdir,
        l=args.l,
        b=args.b,
        n_simu=args.n_simu,
        seed=args.seed,
    )
    if args.stdout_out is not None:
        args.stdout_out.parent.mkdir(parents=True, exist_ok=True)
        args.stdout_out.write_text(stdout)

    raw = parse_verbosity3_stdout(stdout)
    physical, weights, source_group = physical_and_source_group_from_raw(raw)
    resample_size = args.resample_size or args.n_simu
    indices = resample_weighted_indices(physical, weights, resample_size, seed=args.seed + 1009)
    resampled = physical[indices]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        physical=resampled,
        condition=np.broadcast_to(np.asarray([args.l, args.b], dtype=float), (resampled.shape[0], 2)),
        raw_physical=physical,
        raw_weights=weights,
        raw_source_group=source_group,
        source_group=source_group[indices],
        raw_columns=np.asarray(RAW_COLUMNS),
        feature_names=np.asarray(FEATURE_NAMES),
        condition_names=np.asarray(["l", "b"]),
        metadata=json.dumps(
            {
                "source": "genulens CLI stdout",
                "l_deg": float(args.l),
                "b_deg": float(args.b),
                "n_simu": int(args.n_simu),
                "seed": int(args.seed),
                "resample_size": int(resample_size),
                "feature_names": FEATURE_NAMES,
            },
            indent=2,
        ),
    )
    print(f"parsed {raw.shape[0]} weighted events")
    print(f"wrote {resampled.shape[0]} resampled 5D events to {args.out}")
    print("columns:", ", ".join(FEATURE_NAMES))


def run_genulens_cli(
    *,
    executable: Path,
    workdir: Path,
    l: float,
    b: float,
    n_simu: int,
    seed: int,
    remnant: int = 0,
    binary: int = 0,
    nsd: int = 0,
    small_gamma: int = 0,
    source_group: int | None = None,
) -> str:
    executable = executable.resolve()
    workdir = workdir.resolve()
    if source_group is not None and executable.name != "generate_flow_samples":
        executable = executable.parent / "pre_gapmoe" / "generate_flow_samples"
        if not executable.is_file():
            raise FileNotFoundError(
                "source-group flow samples require "
                f"{executable}; build it with `make pre_gapmoe/generate_flow_samples` in {workdir}"
            )
    cmd = [
        str(executable),
        "l",
        str(l),
        "b",
        str(b),
        "Nsimu",
        str(n_simu),
        "seed",
        str(seed),
        "REMNANT",
        str(remnant),
        "BINARY",
        str(binary),
        "NSD",
        str(nsd),
        "SMALLGAMMA",
        str(small_gamma),
        "VERBOSITY",
        "3",
    ]
    if source_group is not None:
        cmd.extend(["SOURCEGROUP", str(source_group)])
    completed = subprocess.run(
        cmd,
        cwd=workdir,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        if source_group is not None and "forced source component has zero density" in completed.stderr:
            raise ZeroSourceDensityError(completed.stderr.strip())
        raise subprocess.CalledProcessError(
            completed.returncode, cmd, output=completed.stdout, stderr=completed.stderr
        )
    if completed.stderr.strip():
        print(completed.stderr, end="")
    return completed.stdout


def parse_verbosity3_stdout(stdout: str) -> np.ndarray:
    rows: list[list[float]] = []
    in_table = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#----- Output of Monte Carlo simulation"):
            in_table = True
            continue
        if not in_table:
            continue
        if stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != len(RAW_COLUMNS):
            continue
        rows.append([float(part) for part in parts])
    if not rows:
        raise ValueError("no VERBOSITY=3 event rows found in genulens stdout.")
    return np.asarray(rows, dtype=float)


def physical_from_raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    physical, weights, _ = physical_and_source_group_from_raw(raw)
    return physical, weights


def physical_and_source_group_from_raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = raw[:, 0]
    ml = raw[:, 1]
    dl = raw[:, 2] / 1000.0
    ds = raw[:, 3] / 1000.0
    pi_e = raw[:, 6]
    pi_en = raw[:, 7]
    pi_ee = raw[:, 8]
    mu_rel = raw[:, 9]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_n = mu_rel * pi_en / pi_e
        mu_e = mu_rel * pi_ee / pi_e
    physical = np.column_stack([ml, dl, ds, mu_e, mu_n])
    valid = valid_raw_event_mask(raw)
    if not np.all(valid):
        physical = physical[valid]
        weights = weights[valid]
    source_group = _source_group(raw[:, 14])[valid]
    if physical.size == 0:
        raise ValueError("no valid physical events after filtering.")
    return physical, weights, source_group


def valid_raw_event_mask(raw: np.ndarray) -> np.ndarray:
    """Rows belonging to the physical support of the conditional Flow."""

    raw = np.asarray(raw, dtype=float)
    weights = raw[:, 0]
    ml = raw[:, 1]
    dl = raw[:, 2] / 1000.0
    ds = raw[:, 3] / 1000.0
    pi_e = raw[:, 6]
    pi_en = raw[:, 7]
    pi_ee = raw[:, 8]
    mu_rel = raw[:, 9]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_n = mu_rel * pi_en / pi_e
        mu_e = mu_rel * pi_ee / pi_e
    physical = np.column_stack([ml, dl, ds, mu_e, mu_n])
    return (
        np.isfinite(physical).all(axis=1)
        & np.isfinite(weights)
        & (weights > 0.0)
        & (ml > 0.0)
        & (dl > 0.0)
        & (ds > dl)
        & (pi_e > 0.0)
    )


def remove_event_rate_weight(physical: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Remove theta_E * mu from SMALLGAMMA=1 genulens weights."""

    ml, dl, ds, mu_e, mu_n = np.asarray(physical, dtype=float).T
    pi_rel_mas = 1.0 / dl - 1.0 / ds
    theta_e_mas = np.sqrt(8.144 * ml * pi_rel_mas)
    event_factor = theta_e_mas * np.hypot(mu_e, mu_n)
    out = np.asarray(weights, dtype=float) / event_factor
    if np.any(~np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError("cannot remove event-rate weight from invalid physical samples")
    return out


def resample_weighted(
    physical: np.ndarray,
    weights: np.ndarray,
    n: int,
    *,
    seed: int,
) -> np.ndarray:
    return physical[resample_weighted_indices(physical, weights, n, seed=seed)]


def resample_weighted_indices(
    physical: np.ndarray,
    weights: np.ndarray,
    n: int,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prob = weights / np.sum(weights)
    return rng.choice(physical.shape[0], size=n, replace=True, p=prob)


def _source_group(component: np.ndarray) -> np.ndarray:
    component = np.asarray(component, dtype=int)
    return np.select(
        [component <= 6, component == 7, component == 8, component == 9],
        [0, 1, 2, 3],
        default=4,
    ).astype(int)


if __name__ == "__main__":
    main()
