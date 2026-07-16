from __future__ import annotations

from math import exp, hypot, isfinite, log
from typing import Any, Callable, Optional

import numpy as np

from gapmoe.density.base import DensityModel
from gapmoe.param_types.base import MappingContext, ParamTypeProtocol
from gapmoe.priors.event_rate import KAPPA, log_event_rate

PhysicalValues = tuple[float, ...]
MuPhysicalValues = tuple[float, float, float, float]
ThetaMuValues = tuple[float, float]
ExtraLogPrior = Callable[[float, float, float, float, float], float]


class _ParameterizedNumpyEngine:
    """Compose a Galactic density model with event-rate and optional mappings.

    Physical parameters use kpc for DL and DS.
    """

    def __init__(
        self,
        density: DensityModel,
        *,
        param_type: Optional[ParamTypeProtocol] = None,
        include_event_rate: bool = True,
        extra_log_prior: Optional[ExtraLogPrior] = None,
    ) -> None:
        self.density = density
        self.param_type = param_type
        self.include_event_rate = include_event_rate
        self.extra_log_prior = extra_log_prior

    def log_prob(self, theta: Any, *args: Any, context: Optional[MappingContext] = None) -> float:
        if len(args) == 1 and isinstance(args[0], dict) and context is None:
            context = args[0]
            args = ()
        if self._marginalizes_distance(args):
            return self._log_prob_marginalized_distance(theta, context)

        if self._uses_theta_mu_physical(args):
            params, log_jacobian = self._to_theta_mu_physical_with_jacobian(theta, context)
            if not isfinite(log_jacobian):
                return float("-inf")
            logp = self.log_prob_theta_mu(params)
            if not isfinite(logp):
                return float("-inf")
            logp += log_jacobian
            if not isfinite(logp):
                return float("-inf")
            return logp

        if self._uses_mu_physical(args):
            params, log_jacobian = self._to_mu_physical_with_jacobian(theta, context)
            if not isfinite(log_jacobian):
                return float("-inf")
            logp = self.log_prob_mu(params)
            if not isfinite(logp):
                return float("-inf")
            logp += log_jacobian
            if not isfinite(logp):
                return float("-inf")
            return logp

        params, log_jacobian = self._to_physical_with_jacobian(theta, args, context)
        if not isfinite(log_jacobian):
            return float("-inf")

        logp = self.log_prob_physical(params)
        if not isfinite(logp):
            return float("-inf")

        if log_jacobian != 0.0:
            logp += log_jacobian
            if not isfinite(logp):
                return float("-inf")

        return logp

    def log_prob_physical(self, values: Any, *args: Any) -> float:
        """Evaluate the Galactic prior directly in physical parameters.

        Parameters are ``(ML, DL, DS, mu_N, mu_E)`` with distances in kpc and
        proper motions in mas/yr. No light-curve param_type Jacobian is
        applied here.
        """
        ML, DL, DS, mu_N, mu_E = _raw_values(values, args)
        mu = hypot(mu_N, mu_E)

        logp = self.density.log_density(ML, DL, DS, mu_N, mu_E)
        if not isfinite(logp):
            return float("-inf")

        if self.include_event_rate:
            logp += log_event_rate(ML, DL, DS, mu)
            if not isfinite(logp):
                return float("-inf")

        if self.extra_log_prior is not None:
            logp += self.extra_log_prior(ML, DL, DS, mu_N, mu_E)
            if not isfinite(logp):
                return float("-inf")

        return logp

    def log_prob_mu(self, values: Any, *args: Any) -> float:
        """Evaluate the prior in ``(ML, DL, DS, mu)`` after marginalizing phi."""
        if self.extra_log_prior is not None:
            raise TypeError("extra_log_prior is not supported for direction-marginalized mu priors.")
        if not hasattr(self.density, "log_density_mu"):
            raise TypeError("density must provide log_density_mu(ML, DL, DS, mu).")

        ML, DL, DS, mu = _raw_mu_values(values, args)
        logp = self.density.log_density_mu(ML, DL, DS, mu)
        if not isfinite(logp):
            return float("-inf")

        if self.include_event_rate:
            logp += log_event_rate(ML, DL, DS, mu)
            if not isfinite(logp):
                return float("-inf")

        return logp

    def log_prob_theta_mu(self, values: Any, *args: Any) -> float:
        """Evaluate the prior in ``(thetaE, mu)`` after marginalizing DL, DS, and phi."""
        if self.extra_log_prior is not None:
            raise TypeError("extra_log_prior is not supported for distance-marginalized priors.")
        if not hasattr(self.density, "log_density_theta_mu"):
            raise TypeError("density must provide log_density_theta_mu(thetaE, mu).")

        theta_e, mu = _raw_theta_mu_values(values, args)
        return float(
            self.density.log_density_theta_mu(
                theta_e,
                mu,
                include_event_rate=self.include_event_rate,
            )
        )

    def to_physical(self, theta: Any, *, context: Optional[MappingContext] = None) -> PhysicalValues:
        """Convert light-curve parameters to physical parameters.

        The first five values are always ``(ML, DL, DS, mu_N, mu_E)``. Param
        types with additional physical/derived coordinates append them after
        those density coordinates.
        """
        if self._uses_theta_mu_physical(()) or self._uses_mu_physical(()):
            raise TypeError(
                "This param_type marginalizes physical dimensions; use "
                "to_theta_mu_physical() or to_mu_physical()."
            )
        if self.param_type is None:
            return _raw_values(theta, ())
        return _coerce_physical(self.param_type.to_physical(theta, context))

    def to_derived(
        self,
        theta: Any,
        *,
        context: Optional[MappingContext] = None,
    ) -> dict[str, float]:
        """Return derived parameters supplied by the param_type.

        Orbital-motion param_types use this to expose orbital elements that are
        computed during the light-curve to physical-parameter transform but are
        not part of the Galactic density coordinates.
        """
        if self.param_type is None or not hasattr(self.param_type, "to_derived"):
            return {}
        return _coerce_derived(self.param_type.to_derived(theta, context))

    def to_mu_physical(
        self,
        theta: Any,
        *,
        context: Optional[MappingContext] = None,
    ) -> MuPhysicalValues:
        """Convert light-curve parameters to ``(ML, DL, DS, mu)``."""
        values, _ = self._to_mu_physical_with_jacobian(theta, context)
        return values

    def to_theta_mu_physical(
        self,
        theta: Any,
        *,
        context: Optional[MappingContext] = None,
    ) -> ThetaMuValues:
        """Convert light-curve parameters to ``(thetaE, mu)``."""
        values, _ = self._to_theta_mu_physical_with_jacobian(theta, context)
        return values

    def log_abs_det_jacobian(
        self,
        theta: Any,
        *,
        context: Optional[MappingContext] = None,
    ) -> float:
        """Return the log absolute Jacobian determinant for ``theta``."""
        if self._marginalizes_distance(()):
            raise TypeError("distance-marginalized param_types integrate the full Jacobian.")
        if self._uses_theta_mu_physical(()):
            _, log_jacobian = self._to_theta_mu_physical_with_jacobian(theta, context)
        elif self._uses_mu_physical(()):
            _, log_jacobian = self._to_mu_physical_with_jacobian(theta, context)
        else:
            _, log_jacobian = self._to_physical_with_jacobian(theta, (), context)
        return log_jacobian

    def sample_physical(
        self,
        theta: Any,
        *,
        context: Optional[MappingContext] = None,
        rng: Any = None,
    ) -> dict[str, float] | dict[str, np.ndarray]:
        """Draw physical parameters conditional on a possibly marginalized theta.

        For distance-marginalized param_types this samples the hidden
        distance variable(s) from the same conditional integrand used by
        ``log_prob``. For ordinary param_types it only converts theta to
        physical parameters. A two-dimensional theta array is treated as a
        collection of theta samples and returns one array per physical
        parameter.
        """
        rng = np.random.default_rng() if rng is None else rng
        theta_array = np.asarray(theta)
        if theta_array.ndim == 2:
            if theta_array.shape[0] == 0:
                raise ValueError("theta array must contain at least one sample.")
            draws = [self._sample_physical_one(row, context, rng) for row in theta_array]
            return _stack_sample_draws(draws)
        if theta_array.ndim > 2:
            raise ValueError("theta must be one-dimensional or two-dimensional.")

        return self._sample_physical_one(theta, context, rng)

    def _sample_physical_one(
        self,
        theta: Any,
        context: Optional[MappingContext],
        rng: Any,
    ) -> dict[str, float]:
        if self._marginalizes_distance(()):
            return self._sample_physical_marginalized_distance(theta, context, rng)
        if self._uses_theta_mu_physical(()):
            return self._sample_physical_theta_mu(theta, context, rng)
        if self._uses_mu_physical(()):
            ML, DL, DS, mu = self.to_mu_physical(theta, context=context)
            return {"ML": ML, "DL": DL, "DS": DS, "mu": mu}

        ML, DL, DS, mu_N, mu_E = self._density_physical(theta, context)
        draw = {"ML": ML, "DL": DL, "DS": DS, "mu_N": mu_N, "mu_E": mu_E}
        if self.param_type is not None:
            full = self.to_physical(theta, context=context)
            derived_names = tuple(getattr(self.param_type, "derived_names", ()))
            draw.update(
                {
                    name: float(value)
                    for name, value in zip(derived_names, full[5:])
                }
            )
        return draw

    def _log_prob_marginalized_distance(
        self,
        theta: Any,
        context: Optional[MappingContext],
    ) -> float:
        if bool(getattr(self.param_type, "supports_distance_grid", False)):
            terms, _ = self._marginalized_distance_log_terms(theta, context)
            return _logsumexp_array(terms)

        distances, weights = _distance_grid_and_weights(self.density)
        param_type = self.param_type
        full_impl = param_type.distance_impl

        log_terms = []
        for distance, weight in zip(distances, weights):
            if weight <= 0.0:
                continue
            full_theta = param_type.with_distance(theta, distance)
            values = _coerce_values(full_impl.to_physical(full_theta, context))
            logp = self.log_prob_physical(values)
            if not isfinite(logp):
                continue
            log_jacobian = full_impl.log_abs_det_jacobian(full_theta, context)
            if not isfinite(log_jacobian):
                continue
            log_terms.append(log(weight) + logp + log_jacobian)
        return _logsumexp(log_terms)

    def _sample_physical_marginalized_distance(
        self,
        theta: Any,
        context: Optional[MappingContext],
        rng: Any,
    ) -> dict[str, float]:
        if bool(getattr(self.param_type, "supports_distance_grid", False)):
            log_terms, physical_values = self._marginalized_distance_log_terms(theta, context)
            idx = _draw_from_log_weights(log_terms, rng)
            ML, DL, DS, mu_N, mu_E = (values[idx] for values in physical_values)
            return {
                "ML": float(ML),
                "DL": float(DL),
                "DS": float(DS),
                "mu_N": float(mu_N),
                "mu_E": float(mu_E),
            }

        distances, weights = _distance_grid_and_weights(self.density)
        param_type = self.param_type
        full_impl = param_type.distance_impl

        log_terms = []
        physical_values = []
        for distance, weight in zip(distances, weights):
            if weight <= 0.0:
                continue
            full_theta = param_type.with_distance(theta, distance)
            values = _coerce_values(full_impl.to_physical(full_theta, context))
            logp = self.log_prob_physical(values)
            if not isfinite(logp):
                continue
            log_jacobian = full_impl.log_abs_det_jacobian(full_theta, context)
            if not isfinite(log_jacobian):
                continue
            log_terms.append(log(weight) + logp + log_jacobian)
            physical_values.append(values)

        idx = _draw_from_log_terms(log_terms, rng)
        ML, DL, DS, mu_N, mu_E = physical_values[idx]
        return {"ML": ML, "DL": DL, "DS": DS, "mu_N": mu_N, "mu_E": mu_E}

    def _marginalized_distance_log_terms(
        self,
        theta: Any,
        context: Optional[MappingContext],
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        distances_list, weights_list = _distance_grid_and_weights(self.density)
        distances = np.asarray(distances_list, dtype=float)
        weights = np.asarray(weights_list, dtype=float)
        physical_values = self.param_type.physical_with_distance_grid(theta, distances, context)
        physical_values = tuple(np.asarray(values, dtype=float) for values in physical_values)
        log_jacobian = np.asarray(
            self.param_type.log_abs_det_jacobian_with_distance_grid(
                theta,
                distances,
                context,
            ),
            dtype=float,
        )

        if hasattr(self.density, "log_density_array") and self.extra_log_prior is None:
            ML, DL, DS, mu_N, mu_E = physical_values
            logp = np.asarray(self.density.log_density_array(ML, DL, DS, mu_N, mu_E), dtype=float)
            if self.include_event_rate:
                logp = logp + _log_event_rate_array(ML, DL, DS, np.hypot(mu_N, mu_E))
        else:
            logp = np.asarray(
                [
                    self.log_prob_physical((ML, DL, DS, mu_N, mu_E))
                    for ML, DL, DS, mu_N, mu_E in zip(*physical_values)
                ],
                dtype=float,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            log_terms = np.log(weights) + logp + log_jacobian
        log_terms = np.where(np.isfinite(log_terms), log_terms, -np.inf)
        return log_terms, physical_values

    def _sample_physical_theta_mu(
        self,
        theta: Any,
        context: Optional[MappingContext],
        rng: Any,
    ) -> dict[str, float]:
        theta_e, mu = self.to_theta_mu_physical(theta, context=context)
        grid = self._distance_conditional_grid(theta_e, mu)
        flat_weights = grid["weights"].ravel()
        total = float(np.sum(flat_weights))
        if total <= 0.0 or not isfinite(total):
            raise ValueError("conditional distance distribution has zero probability.")
        idx = int(rng.choice(flat_weights.size, p=flat_weights / total))
        shape = grid["weights"].shape
        i, j = np.unravel_index(idx, shape)
        DL = float(grid["dl"][i, j])
        DS = float(grid["ds"][i, j])
        ML = float(grid["mass"][i, j])
        return {"ML": ML, "DL": DL, "DS": DS, "mu": float(mu)}

    def _distance_conditional_grid(self, theta_e: float, mu: float) -> dict[str, np.ndarray]:
        theta_e = float(theta_e)
        mu = float(mu)
        if hasattr(self.density, "_distance_grid"):
            grid = self.density._distance_grid()
            safe_pi_rel = np.where(grid.valid, grid.pi_rel, 1.0)
            mass = theta_e * theta_e / (KAPPA * safe_pi_rel)
            jac = 2.0 * theta_e / (KAPPA * safe_pi_rel)
            p_mass = self.density._mass_density_grid(mass, grid.component_fractions)
            p_mu = self.density.murel.mu_density_for_pair_indices(grid.pair_indices, mu)
            weights = grid.weights * p_mass * p_mu * jac
            if self.include_event_rate:
                weights *= grid.dl * grid.dl * theta_e * mu
            weights = np.where(grid.valid, weights, 0.0)
            return {"dl": grid.dl, "ds": grid.ds, "mass": mass, "weights": weights}

        distances_list, weights_list = _distance_grid_and_weights(self.density)
        distances = np.asarray(distances_list, dtype=float)
        one_d_weights = np.asarray(weights_list, dtype=float)
        dl, ds = np.meshgrid(distances, distances, indexing="ij")
        dl_weights, ds_weights = np.meshgrid(one_d_weights, one_d_weights, indexing="ij")
        valid = dl < ds
        pi_rel = np.where(valid, (1.0 / dl) - (1.0 / ds), 1.0)
        mass = theta_e * theta_e / (KAPPA * pi_rel)
        jac = 2.0 * theta_e / (KAPPA * pi_rel)
        weights = np.zeros_like(dl, dtype=float)
        for index in np.ndindex(dl.shape):
            if not valid[index]:
                continue
            logp = self.log_prob_mu((mass[index], dl[index], ds[index], mu))
            if isfinite(logp):
                weights[index] = exp(logp) * jac[index] * dl_weights[index] * ds_weights[index]
        return {"dl": dl, "ds": ds, "mass": mass, "weights": weights}

    def _to_physical_with_jacobian(
        self,
        theta: Any,
        args: tuple[Any, ...],
        context: Optional[MappingContext],
    ) -> tuple[PhysicalValues, float]:
        if self.param_type is None:
            return _raw_values(theta, args), 0.0

        if args:
            raise TypeError("a parameterized galaxy expects one theta object, not raw arguments.")
        values = _coerce_values(self.param_type.to_physical(theta, context))
        log_jacobian = self.param_type.log_abs_det_jacobian(theta, context)
        return values, log_jacobian

    def _density_physical(
        self,
        theta: Any,
        context: Optional[MappingContext],
    ) -> tuple[float, float, float, float, float]:
        values, _ = self._to_physical_with_jacobian(theta, (), context)
        return values

    def _to_mu_physical_with_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext],
    ) -> tuple[MuPhysicalValues, float]:
        if self.param_type is None or not hasattr(self.param_type, "to_mu_physical"):
            return _raw_mu_values(theta, ()), 0.0

        values = _coerce_mu_values(self.param_type.to_mu_physical(theta, context))
        log_jacobian = self.param_type.log_abs_det_jacobian(theta, context)
        return values, log_jacobian

    def _to_theta_mu_physical_with_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext],
    ) -> tuple[ThetaMuValues, float]:
        if self.param_type is None or not hasattr(self.param_type, "to_theta_mu_physical"):
            return _raw_theta_mu_values(theta, ()), 0.0

        values = _coerce_theta_mu_values(self.param_type.to_theta_mu_physical(theta, context))
        log_jacobian = self.param_type.log_abs_det_jacobian(theta, context)
        return values, log_jacobian

    def _uses_theta_mu_physical(self, args: tuple[Any, ...]) -> bool:
        return (
            self.param_type is not None
            and not args
            and bool(getattr(self.param_type, "uses_theta_mu_physical", False))
        )

    def _marginalizes_distance(self, args: tuple[Any, ...]) -> bool:
        return (
            self.param_type is not None
            and not args
            and bool(getattr(self.param_type, "marginalizes_distance", False))
        )

    def _uses_mu_physical(self, args: tuple[Any, ...]) -> bool:
        return (
            self.param_type is not None
            and not args
            and bool(getattr(self.param_type, "uses_mu_physical", False))
        )


def _raw_values(first: Any, rest: tuple[Any, ...]) -> PhysicalValues:
    if rest:
        return _coerce_values((first, *rest))
    return _coerce_values(first)


def _coerce_values(values: Any) -> tuple[float, float, float, float, float]:
    if len(values) < 5:
        raise TypeError("log_prob expects at least ML, DL, DS, mu_N, mu_E.")
    ML, DL, DS, mu_N, mu_E = values[:5]
    return float(ML), float(DL), float(DS), float(mu_N), float(mu_E)


def _coerce_physical(values: Any) -> PhysicalValues:
    if len(values) < 5:
        raise TypeError("to_physical expects at least ML, DL, DS, mu_N, mu_E.")
    return tuple(float(value) for value in values)


def _raw_mu_values(first: Any, rest: tuple[Any, ...]) -> MuPhysicalValues:
    if rest:
        return _coerce_mu_values((first, *rest))
    return _coerce_mu_values(first)


def _coerce_mu_values(values: Any) -> MuPhysicalValues:
    if len(values) != 4:
        raise TypeError("log_prob_mu expects ML, DL, DS, mu.")
    ML, DL, DS, mu = values
    return float(ML), float(DL), float(DS), float(mu)


def _raw_theta_mu_values(first: Any, rest: tuple[Any, ...]) -> ThetaMuValues:
    if rest:
        return _coerce_theta_mu_values((first, *rest))
    return _coerce_theta_mu_values(first)


def _coerce_theta_mu_values(values: Any) -> ThetaMuValues:
    if len(values) != 2:
        raise TypeError("log_prob_theta_mu expects thetaE, mu.")
    theta_e, mu = values
    return float(theta_e), float(mu)


def _coerce_derived(values: Any) -> dict[str, float]:
    return {str(key): float(value) for key, value in dict(values).items()}


def _stack_sample_draws(draws: list[dict[str, float]]) -> dict[str, np.ndarray]:
    keys = draws[0].keys()
    return {key: np.asarray([draw[key] for draw in draws], dtype=float) for key in keys}


def _log_event_rate_array(
    ML: np.ndarray,
    DL: np.ndarray,
    DS: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        pi_rel = (1.0 / DL) - (1.0 / DS)
        theta_e = np.sqrt(ML * pi_rel * KAPPA)
        log_rate = 2.0 * np.log(DL) + np.log(theta_e) + np.log(mu)
    valid = (
        (ML > 0.0)
        & (DL > 0.0)
        & (DS > DL)
        & (mu > 0.0)
        & (pi_rel > 0.0)
        & (theta_e > 0.0)
        & np.isfinite(theta_e)
    )
    return np.where(valid & np.isfinite(log_rate), log_rate, -np.inf)


def _distance_grid_and_weights(density: Any) -> tuple[list[float], list[float]]:
    if not hasattr(density, "distance") or not hasattr(density.distance, "distance_pc"):
        raise TypeError("density must expose distance.distance_pc for distance marginalization.")
    distances = [float(x) / 1000.0 for x in density.distance.distance_pc]
    if len(distances) < 2:
        raise ValueError("distance grid must contain at least two points.")

    weights = []
    for i, distance in enumerate(distances):
        if i == 0:
            weights.append(0.5 * (distances[1] - distance))
        elif i == len(distances) - 1:
            weights.append(0.5 * (distance - distances[i - 1]))
        else:
            weights.append(0.5 * (distances[i + 1] - distances[i - 1]))
    return distances, weights


def _logsumexp(log_terms: list[float]) -> float:
    if not log_terms:
        return float("-inf")
    max_log = max(log_terms)
    if not isfinite(max_log):
        return float("-inf")
    return max_log + log(sum(exp(value - max_log) for value in log_terms))


def _logsumexp_array(log_terms: np.ndarray) -> float:
    if log_terms.size == 0:
        return float("-inf")
    max_log = float(np.max(log_terms))
    if not isfinite(max_log):
        return float("-inf")
    return float(max_log + np.log(np.sum(np.exp(log_terms - max_log))))


def _draw_from_log_terms(log_terms: list[float], rng: Any) -> int:
    if not log_terms:
        raise ValueError("conditional distribution has zero probability.")
    max_log = max(log_terms)
    weights = np.asarray([exp(value - max_log) for value in log_terms], dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0 or not isfinite(total):
        raise ValueError("conditional distribution has zero probability.")
    return int(rng.choice(len(weights), p=weights / total))


def _draw_from_log_weights(log_terms: np.ndarray, rng: Any) -> int:
    if log_terms.size == 0:
        raise ValueError("conditional distribution has zero probability.")
    max_log = float(np.max(log_terms))
    if not isfinite(max_log):
        raise ValueError("conditional distribution has zero probability.")
    weights = np.exp(log_terms - max_log)
    total = float(np.sum(weights))
    if total <= 0.0 or not isfinite(total):
        raise ValueError("conditional distribution has zero probability.")
    return int(rng.choice(weights.size, p=weights / total))
