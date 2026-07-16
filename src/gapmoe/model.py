"""Complete, sampler-independent Galactic inference models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .flow_package import FlowPackage
from .flow_releases import get_flow_release
from .priors.high_level import GalaxyModel, IsochroneModel as Isochrone
from .priors.parameterized import ParameterizedGalaxyModel
from .pre_runner import PreRunResult


@dataclass(frozen=True)
class Flow:
    """A trained Flow backend used to construct a physical Galaxy model."""

    release: str = "rate-included-v1"
    package: FlowPackage | None = None

    def build(
        self,
        *,
        source: Isochrone,
        l: float,
        b: float,
        extinction: Mapping[str, float],
        dm_rc: float | None,
        dust_scale_height_pc: float,
        include_event_rate: bool,
        remnant: int,
        binary: int,
    ) -> GalaxyModel:
        release = get_flow_release(self.release)
        release.validate_sightline(l, b)
        release.validate_model_options(remnant=remnant, binary=binary)
        if release.event_rate_included and not include_event_rate:
            raise ValueError(
                f"flow release {release.name!r} is trained on the event-rate measure; "
                "include_event_rate=False cannot remove that factor"
            )
        package = self.package or FlowPackage.bundled(release.name)
        if package.manifest.release != release.name:
            raise ValueError(
                f"Flow package release {package.manifest.release!r} does not match "
                f"the requested release {release.name!r}"
            )
        return GalaxyModel.from_flow_package(
            package,
            isochrone=source,
            l_deg=float(l),
            b_deg=float(b),
            extinction_at_rc=extinction,
            dm_rc=dm_rc,
            dust_scale_height_pc=dust_scale_height_pc,
            include_event_rate=include_event_rate,
        )


@dataclass(frozen=True)
class Histogram:
    """A precomputed event-local histogram backend."""

    pre_run: PreRunResult

    @classmethod
    def open(cls, directory: str | Path) -> "Histogram":
        directory = Path(directory).expanduser().resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        paths = {
            name: _artifact_path(directory, manifest[f"{name}_path"])
            for name in ("mass", "rho", "murel")
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"incomplete gapmoe histogram: missing {', '.join(missing)}"
            )
        source_path = manifest.get("source_evidence_path")
        cmd_path = manifest.get("cmd_prior_path")
        return cls(
            PreRunResult(
                ra_deg=manifest.get("ra_deg"),
                dec_deg=manifest.get("dec_deg"),
                l_deg=float(manifest["l_deg"]),
                b_deg=float(manifest["b_deg"]),
                output_dir=directory,
                mass_path=paths["mass"],
                rho_path=paths["rho"],
                murel_path=paths["murel"],
                manifest_path=directory / "manifest.json",
                source_evidence_path=(
                    _artifact_path(directory, source_path) if source_path else None
                ),
                cmd_prior_path=(
                    _artifact_path(directory, cmd_path) if cmd_path else None
                ),
                commands=manifest.get("commands", {}),
            )
        )

    def build(
        self,
        *,
        source: Isochrone,
        l: float,
        b: float,
        extinction: Mapping[str, float],
        dm_rc: float | None,
        dust_scale_height_pc: float,
        include_event_rate: bool,
        remnant: int,
        binary: int,
    ) -> GalaxyModel:
        remnant = _binary_model_option("remnant", remnant)
        binary = _binary_model_option("binary", binary)
        _validate_prepared_model_option(
            "remnant",
            remnant,
            _prepared_model_option(self.pre_run, "REMNANT"),
        )
        _validate_prepared_model_option(
            "binary",
            binary,
            _prepared_model_option(self.pre_run, "BINARY"),
        )
        if not _same_sightline(l, self.pre_run.l_deg) or not _same_sightline(
            b, self.pre_run.b_deg
        ):
            raise ValueError(
                "Histogram sightline disagrees with Model(l=..., b=...)"
            )
        return GalaxyModel.from_pre_run(
            self.pre_run,
            isochrone=source,
            extinction_at_rc=extinction,
            dm_rc=dm_rc,
            dust_scale_height_pc=dust_scale_height_pc,
            include_event_rate=include_event_rate,
        )


class Model:
    """A complete Galactic prior expressed in user-selected coordinates.

    The model is independent of any sampler or light-curve package.  It owns
    the physical density, parameter transform, Jacobian, and any hidden-variable
    integration required by ``param_type``.
    """

    def __init__(
        self,
        param_type: Any,
        *,
        l: float,
        b: float,
        source: Isochrone,
        extinction: Mapping[str, float] | None = None,
        backend: Any | None = None,
        dm_rc: float | None = None,
        dust_scale_height_pc: float = 164.0,
        include_event_rate: bool = True,
        remnant: int = 0,
        binary: int = 0,
        integration_samples: int = 512,
        direction_samples: int = 32,
        seed: int = 0,
    ) -> None:
        if source.table is None:
            source = source.build(
                reference_edges=_default_reference_edges(),
                color_edges=_default_color_edges(),
            )
        backend = Flow() if backend is None else backend
        build = getattr(backend, "build", None)
        if build is None:
            raise TypeError(
                "backend must provide build(source=..., l=..., b=..., extinction=..., ...)"
            )
        physical = build(
            source=source,
            l=float(l),
            b=float(b),
            extinction={
                str(name): float(value)
                for name, value in ({} if extinction is None else extinction).items()
            },
            dm_rc=None if dm_rc is None else float(dm_rc),
            dust_scale_height_pc=float(dust_scale_height_pc),
            include_event_rate=bool(include_event_rate),
            remnant=int(remnant),
            binary=int(binary),
        )
        self.physical = physical
        self.source = source
        self.isochrone = source
        self.backend = backend
        self.param_type = param_type
        self._model = ParameterizedGalaxyModel(
            physical,
            param_type,
            integration_samples=integration_samples,
            direction_samples=direction_samples,
            seed=seed,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return self._model.names

    @property
    def integration_samples(self) -> int:
        return self._model.integration_samples

    @property
    def direction_samples(self) -> int:
        return self._model.direction_samples

    def prior(self, fn):
        """Add a JAX-compatible prior over physical or derived quantities."""

        return self._model.prior(fn)

    def log_density(self, theta, *, context=None, magnitudes=None):
        return self._model.log_density(theta, context=context, magnitudes=magnitudes)

    def log_joint_density(self, theta, *, magnitudes, context=None):
        return self._model.log_joint_density(theta, magnitudes=magnitudes, context=context)

    def log_density_batch(self, theta, *, context=None, magnitudes=None, joint=False):
        return self._model.log_density_batch(
            theta,
            context=context,
            magnitudes=magnitudes,
            joint=joint,
        )

    def is_valid(self, theta, *, context=None):
        return self._model.is_valid(theta, context=context)

    def to_physical(self, theta, *, context=None):
        return self._model.to_physical(theta, context=context)

    def to_deterministic_physical(self, theta, *, context=None):
        return self._model.to_deterministic_physical(theta, context=context)

    def to_derived(self, theta, *, context=None):
        return self._model.to_derived(theta, context=context)

    def log_abs_det_jacobian(self, theta, *, context=None):
        return self._model.log_abs_det_jacobian(theta, context=context)

    def sample_physical(
        self,
        theta,
        *,
        context=None,
        magnitudes=None,
        joint=False,
        rng=None,
    ):
        return self._model.sample_physical(
            theta,
            context=context,
            magnitudes=magnitudes,
            joint=joint,
            rng=rng,
        )

    def log_source_density(self, *, ds, magnitudes, context=None):
        return self.physical.log_source_density(
            ds=ds,
            magnitudes=magnitudes,
            context=context,
        )

    def source_radius(self, *, ds, magnitudes, context=None):
        return self.physical.source_radius(
            ds=ds,
            magnitudes=magnitudes,
            context=context,
        )

    def log_theta_star_density(
        self,
        *,
        theta_star_mas,
        ds,
        magnitudes,
        context=None,
    ):
        return self.physical.log_theta_star_density(
            theta_star_mas=theta_star_mas,
            ds=ds,
            magnitudes=magnitudes,
            context=context,
        )

    # Small private protocol used by consumers that jointly integrate source
    # photometry, theta-star, and hidden Galactic variables.
    def _isochrone_conditional_terms(self, theta, *, magnitudes, context=None):
        """Return QMC terms for ``p(physical | magnitudes)``.

        The marginal CMD factor ``log p(magnitudes)`` is excluded. Consumers
        requiring a joint target must add it separately or call the public
        :meth:`log_joint_density` method.
        """

        return self._model._isochrone_conditional_terms(
            theta,
            magnitudes=magnitudes,
            context=context,
        )


def _default_reference_edges():
    import numpy as np

    return np.linspace(-8.0, 20.0, 561)


def _default_color_edges():
    import numpy as np

    return np.linspace(-2.0, 8.0, 201)


def _same_sightline(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1.0e-10


def _artifact_path(directory: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else directory / path


def _binary_model_option(name: str, value: int) -> int:
    if value not in (0, 1, False, True):
        raise ValueError(f"{name} must be 0 or 1")
    return int(value)


def _prepared_model_option(pre_run: PreRunResult, option: str) -> int | None:
    values = set()
    for command in pre_run.commands.values():
        tokens = [str(token) for token in command]
        for index, token in enumerate(tokens[:-1]):
            if token.upper() == option:
                values.add(
                    _binary_model_option(option.lower(), int(tokens[index + 1]))
                )
    if len(values) > 1:
        raise ValueError(f"Histogram manifest disagrees on prepared {option}")
    return next(iter(values), None)


def _validate_prepared_model_option(
    name: str,
    requested: int,
    prepared: int | None,
) -> None:
    if prepared is None:
        if requested != 0:
            raise ValueError(
                f"Histogram metadata does not record {name}; "
                f"cannot use {name}={requested}"
            )
        return
    if requested != prepared:
        raise ValueError(
            f"Histogram was prepared with {name}={prepared}, not {requested}"
        )


__all__ = ["Flow", "Histogram", "Isochrone", "Model"]
