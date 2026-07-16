from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flowjax.train import fit_to_data

from selection_aware_flow import (
    CONDITION_NAMES,
    ConditionTransform,
    FlowConfig,
    ResidualFlowDensity,
    ResidualTransform,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train p(ML,DL,mu_E,mu_N | l,b,DS,source-group).")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--init-model-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow-layers", type=int, default=10)
    parser.add_argument("--nn-width", type=int, default=96)
    parser.add_argument("--nn-depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Deterministic training subset for convergence smoke tests.",
    )
    args = parser.parse_args()

    data = np.load(args.samples, allow_pickle=False)
    rest, condition = np.asarray(data["rest"], dtype=float), np.asarray(data["condition"], dtype=float)
    valid = ResidualTransform.valid_mask_np(rest, condition)
    rest, condition = rest[valid], condition[valid]
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples must be positive")
        if len(rest) > args.max_samples:
            rng = np.random.default_rng(args.seed)
            selected = rng.choice(len(rest), size=args.max_samples, replace=False)
            rest, condition = rest[selected], condition[selected]
    if args.init_model_dir is None:
        transform = ResidualTransform.fit(rest, condition)
        condition_transform = ConditionTransform.fit(condition, names=CONDITION_NAMES)
        config = FlowConfig(dim=4, cond_dim=condition.shape[1], flow_layers=args.flow_layers, nn_width=args.nn_width, nn_depth=args.nn_depth, seed=args.seed)
        density = ResidualFlowDensity.init(transform, config, condition_transform)
    else:
        density = ResidualFlowDensity.load(args.init_model_dir)
        transform = density.transform
        condition_transform = density.condition_transform
        config = density.config
        if config.dim != 4 or config.cond_dim != condition.shape[1]:
            raise ValueError("initial flow dimensions do not match the residual-flow table")
    train_values = jnp.asarray(transform.to_unconstrained_np(rest, condition))
    train_condition = jnp.asarray(condition_transform.to_standard_np(condition))
    if not np.isfinite(np.asarray(train_values)).all() or not np.isfinite(np.asarray(train_condition)).all():
        raise ValueError("non-finite transformed training data")
    transformed_abs = np.abs(np.asarray(train_values))
    print(
        "preflight: "
        f"n={len(rest)} max_abs_z={transformed_abs.max(axis=0).tolist()}",
        flush=True,
    )
    check_n = min(len(rest), 65_536)
    initial_log_prob = np.asarray(jax.block_until_ready(
        density.flow.log_prob(
            train_values[:check_n], condition=train_condition[:check_n]
        )
    ))
    if not np.isfinite(initial_log_prob).all():
        raise FloatingPointError("non-finite initial flow log probability")
    print(
        "preflight: initial_log_prob "
        f"min={initial_log_prob.min():.6g} median={np.median(initial_log_prob):.6g} "
        f"max={initial_log_prob.max():.6g}",
        flush=True,
    )
    if args.gradient_clip <= 0.0:
        raise ValueError("gradient-clip must be positive")
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip),
        optax.adam(args.learning_rate),
    )
    flow, losses = fit_to_data(
        jax.random.key(args.seed + 1), density.flow, data=(train_values, train_condition),
        optimizer=optimizer, max_epochs=args.epochs, max_patience=args.patience,
        batch_size=args.batch_size, return_best=True,
    )
    loss_leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(losses)]
    if not all(np.isfinite(leaf).all() for leaf in loss_leaves):
        raise FloatingPointError("training produced a non-finite loss; refusing to save release artifact")
    trained = ResidualFlowDensity(flow, transform, config, condition_transform)
    trained.save(args.out_dir)
    if isinstance(losses, dict):
        np.savez_compressed(
            args.out_dir / "losses.npz",
            **{name: np.asarray(values, dtype=float) for name, values in losses.items()},
        )
    else:
        np.savez_compressed(args.out_dir / "losses.npz", losses=np.asarray(losses, dtype=float))
    print(f"saved selection-aware flow to {args.out_dir}")


if __name__ == "__main__":
    main()
