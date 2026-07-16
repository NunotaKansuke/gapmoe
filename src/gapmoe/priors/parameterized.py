"""Light-curve parameterization of a physical :class:`GalaxyModel`."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, ndtri
import numpy as np

from .event_rate_backend import log_event_rate_backend, log_flow_kernel_rate_backend
from .galactic import _ParameterizedNumpyEngine
from .galactic_jax import _ParameterizedJaxEngine


_KAPPA = 8.1429


def _has_physical_sampler(density):
    return callable(getattr(density, "sample_source_group", None)) and callable(
        getattr(density, "_sample_kernel", None)
    )


@dataclass(frozen=True)
class _IntegrationProposal:
    ds: Any
    phi: Any
    log_q_ds: Any
    theta_u: Any


@dataclass(frozen=True)
class _MassImportanceProposal(_IntegrationProposal):
    mass: Any
    log_q_mass: Any
    source_group: Any
    log_q_source_group: Any


def _mass_proposal_geometry(proposal, theta_e, mu):
    ds = proposal.ds
    ml = proposal.mass
    x = _KAPPA * ml / (_KAPPA * ml + theta_e**2 * ds)
    dl = x * ds
    log_jacobian = (
        jnp.log(2.0 * x * ds * (1.0 - x))
        + jnp.log(jnp.abs(mu))
        - jnp.log(jnp.abs(theta_e))
    )
    return ml, dl, ds, log_jacobian


@dataclass(frozen=True)
class _PhysicalDensityView:
    galaxy: Any
    magnitudes: Mapping[str, Any] | None = None
    joint: bool = False
    context: Mapping[str, Any] | None = None
    physical_priors: tuple[Any, ...] = ()
    proposal: _IntegrationProposal | None = None
    direction_phi: Any = None
    source_group_qmc: bool = False

    @property
    def _prior(self):
        return self.galaxy._conditional_prior if self.magnitudes is not None else self.galaxy._selected_prior

    @property
    def distance(self):
        return self._prior.density.distance

    def log_density(self, ml, dl, ds, mu_n, mu_e):
        physical = (ml, dl, ds, mu_n, mu_e)
        if self.joint:
            value = self.galaxy.log_joint_density(
                physical,
                magnitudes=self.magnitudes,
                context=self.context,
            )
        else:
            value = self.galaxy.log_density(
                physical,
                magnitudes=self.magnitudes,
                context=self.context,
            )
        return value + _physical_prior_sum(
            self.physical_priors,
            ML=ml,
            DL=dl,
            DS=ds,
            mu_N=mu_n,
            mu_E=mu_e,
            mu=jnp.hypot(mu_n, mu_e),
        )

    def log_density_array(self, ml, dl, ds, mu_n, mu_e):
        return jax.vmap(self.log_density)(ml, dl, ds, mu_n, mu_e)

    def log_density_mu(self, ml, dl, ds, mu):
        density = self._prior.density
        evaluator = getattr(density, "log_density_mu", None)
        if evaluator is not None and self.magnitudes is None and not self.physical_priors:
            value = evaluator(ml, dl, ds, mu)
            if self._prior.include_event_rate and not getattr(density, "event_rate_included", False):
                value = value + self._event_rate(ml, dl, ds, mu)
            return value
        terms, _ = self.direction_log_terms(ml, dl, ds, mu)
        return logsumexp(terms) - jnp.log(terms.shape[0])

    def log_density_theta_mu(self, theta_e, mu, *, include_event_rate=False):
        density = self._prior.density
        evaluator = getattr(density, "log_density_theta_mu", None)
        if evaluator is not None and self.magnitudes is None and not self.physical_priors:
            return evaluator(
                theta_e,
                mu,
                include_event_rate=(
                    self._prior.include_event_rate
                    and not getattr(density, "event_rate_included", False)
                ),
            )
        terms, _ = self.theta_mu_log_terms(theta_e, mu)
        return logsumexp(terms) - jnp.log(terms.shape[0])

    def direction_log_terms(self, ml, dl, ds, mu):
        if self.direction_phi is None:
            raise RuntimeError("proper-motion direction proposal is unavailable")
        phi = self.direction_phi
        mu_n = mu * jnp.cos(phi)
        mu_e = mu * jnp.sin(phi)
        logp = jax.vmap(self.log_density)(
            jnp.full_like(phi, ml),
            jnp.full_like(phi, dl),
            jnp.full_like(phi, ds),
            mu_n,
            mu_e,
        )
        terms = logp + jnp.log(jnp.abs(mu)) + jnp.log(2.0 * jnp.pi)
        return terms, (mu_n, mu_e)

    def theta_mu_log_terms(self, theta_e, mu):
        if not isinstance(self.proposal, _MassImportanceProposal):
            raise RuntimeError("mass importance proposal is unavailable")
        ml, dl, ds, log_jacobian = _mass_proposal_geometry(
            self.proposal, theta_e, mu
        )
        mu_n = mu * jnp.cos(self.proposal.phi)
        mu_e = mu * jnp.sin(self.proposal.phi)
        if self.source_group_qmc:
            density = self._prior.density
            logp = jax.vmap(density.log_density_source_group)(
                ml,
                dl,
                ds,
                mu_n,
                mu_e,
                self.proposal.source_group,
            )
            if (
                self._prior.include_event_rate
                and not getattr(density, "event_rate_included", False)
            ):
                logp = logp + self._event_rate(ml, dl, ds, mu)
            logp = logp + jax.vmap(
                lambda current_ml, current_dl, current_ds, current_mu_n, current_mu_e: _physical_prior_sum(
                    self.physical_priors,
                    ML=current_ml,
                    DL=current_dl,
                    DS=current_ds,
                    mu_N=current_mu_n,
                    mu_E=current_mu_e,
                    mu=jnp.hypot(current_mu_n, current_mu_e),
                )
            )(ml, dl, ds, mu_n, mu_e)
        else:
            logp = jax.vmap(self.log_density)(ml, dl, ds, mu_n, mu_e)
        log_q = (
            self.proposal.log_q_ds
            + self.proposal.log_q_mass
            - jnp.log(2.0 * jnp.pi)
        )
        if self.source_group_qmc:
            log_q = log_q + self.proposal.log_q_source_group
        terms = logp + log_jacobian - log_q
        valid = (
            (theta_e > 0.0)
            & (mu > 0.0)
            & (dl > 0.0)
            & (dl < ds)
            & (ml > 0.0)
        )
        terms = jnp.where(valid & jnp.isfinite(terms), terms, -jnp.inf)
        return terms, (ml, dl, ds, mu_n, mu_e)

    def _event_rate(self, ml, dl, ds, mu):
        density = self._prior.density
        if getattr(density, "event_rate_factor_includes_lens_area", False):
            return log_flow_kernel_rate_backend(ml, dl, ds, mu)
        return log_event_rate_backend(ml, dl, ds, mu)


@dataclass
class ParameterizedGalaxyModel:
    """A physical Galaxy model expressed in light-curve parameters.

    The object is built independently by gapmoe and can be passed to an
    inference package through the small ``names``/``log_density`` protocol.
    """

    galaxy: Any
    param_type: Any
    integration_samples: int = 512
    direction_samples: int = 32
    source_group_integration: str = "exact"
    seed: int = 0
    _physical_priors: list[Any] = field(default_factory=list, init=False, repr=False)
    _proposal: _IntegrationProposal | None = field(default=None, init=False, repr=False)
    _compiled: dict[Any, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        names = getattr(self.param_type, "names", None)
        if names is None or not tuple(names):
            raise TypeError("param_type must expose a non-empty names sequence")
        for name, value in (
            ("integration_samples", self.integration_samples),
            ("direction_samples", self.direction_samples),
        ):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.source_group_integration not in {"exact", "qmc"}:
            raise ValueError(
                "source_group_integration must be 'exact' or 'qmc'"
            )
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.source_group_integration == "qmc":
            if not bool(getattr(self.param_type, "uses_theta_mu_physical", False)):
                raise ValueError(
                    "source_group_integration='qmc' is available only for "
                    "parallax-free distance-marginalized parameterizations"
                )
            if not callable(
                getattr(self.galaxy.density, "log_density_source_group", None)
            ):
                raise TypeError(
                    "source_group_integration='qmc' requires the Flow backend"
                )
        if (
            bool(getattr(self.param_type, "uses_theta_mu_physical", False))
            and _has_physical_sampler(self.galaxy.density)
        ):
            self._importance_proposal()

    @property
    def names(self):
        return tuple(self.param_type.names)

    def prior(self, fn):
        """Add a JAX-compatible prior inside hidden-physical integration."""

        if not callable(fn):
            raise TypeError("parameterized Galaxy prior expects a callable")
        self._physical_priors.append(fn)
        self._compiled.clear()
        return fn

    def _view(self, magnitudes=None, *, joint=False, context=None):
        if joint and magnitudes is None:
            raise ValueError("joint source photometry requires magnitudes")
        density = (
            self.galaxy._conditional_prior.density
            if magnitudes is not None
            else self.galaxy._selected_prior.density
        )
        needs_distance_qmc = bool(
            getattr(self.param_type, "uses_theta_mu_physical", False)
        ) and (
            magnitudes is not None
            or bool(self._physical_priors)
            or not hasattr(density, "log_density_theta_mu")
        )
        needs_direction_qmc = bool(
            getattr(self.param_type, "uses_mu_physical", False)
        ) and (
            magnitudes is not None
            or bool(self._physical_priors)
            or not hasattr(density, "log_density_mu")
        )
        proposal = self._importance_proposal() if needs_distance_qmc else None
        direction_phi = (
            2.0
            * jnp.pi
            * (jnp.arange(self.direction_samples) + 0.5)
            / self.direction_samples
            if needs_direction_qmc
            else None
        )
        return _PhysicalDensityView(
            self.galaxy,
            magnitudes,
            joint,
            context,
            tuple(self._physical_priors),
            proposal,
            direction_phi,
            source_group_qmc=(
                self.source_group_integration == "qmc"
                and magnitudes is None
                and proposal is not None
            ),
        )

    def _importance_proposal(self):
        if (
            bool(getattr(self.param_type, "uses_theta_mu_physical", False))
            and not _has_physical_sampler(self.galaxy.density)
        ):
            raise RuntimeError(
                "this backend cannot importance-sample hidden ML, DL, and DS "
                "for a parallax-free model with dynamic source conditioning. "
                "Use the Flow backend, remove the dynamic magnitudes/physical "
                "prior, or sample the distances explicitly."
            )
        if self._proposal is None:
            self._proposal = _build_integration_proposal(
                self.galaxy,
                self.integration_samples,
                self.seed,
                with_mass=bool(
                    getattr(self.param_type, "uses_theta_mu_physical", False)
                ),
            )
        return self._proposal

    def _jax_engine(self, magnitudes=None, *, joint=False, context=None):
        return _ParameterizedJaxEngine(
            self._view(magnitudes, joint=joint, context=context),
            param_type=self.param_type,
            include_event_rate=False,
        )

    def _numpy_engine(self, magnitudes=None, *, joint=False, context=None):
        return _ParameterizedNumpyEngine(
            self._view(magnitudes, joint=joint, context=context),
            param_type=self.param_type,
            include_event_rate=False,
        )

    def log_density(self, theta, *, context=None, magnitudes=None):
        self._prepare_integration(magnitudes)
        evaluator = self._compiled_evaluator(
            joint=False,
            has_context=context is not None,
            has_magnitudes=magnitudes is not None,
        )
        args = [jnp.asarray(theta)]
        if context is not None:
            args.append(context)
        if magnitudes is not None:
            args.append(magnitudes)
        return evaluator(*args)

    def is_valid(self, theta, *, context=None):
        """Return whether a direct parameter transform is physically valid."""

        if bool(getattr(self.param_type, "marginalizes_distance", False)):
            return True
        value = self.param_type.log_abs_det_jacobian(jnp.asarray(theta), context)
        return jnp.isfinite(value)

    def log_joint_density(self, theta, *, magnitudes, context=None):
        self._prepare_integration(magnitudes)
        evaluator = self._compiled_evaluator(
            joint=True,
            has_context=context is not None,
            has_magnitudes=True,
        )
        args = [jnp.asarray(theta)]
        if context is not None:
            args.append(context)
        args.append(magnitudes)
        return evaluator(*args)

    def _prepare_integration(self, magnitudes):
        density = (
            self.galaxy._conditional_prior.density
            if magnitudes is not None
            else self.galaxy._selected_prior.density
        )
        if bool(getattr(self.param_type, "uses_theta_mu_physical", False)) and (
            magnitudes is not None
            or bool(self._physical_priors)
            or not hasattr(density, "log_density_theta_mu")
        ):
            self._importance_proposal()

    def _compiled_evaluator(self, *, joint, has_context, has_magnitudes):
        key = (joint, has_context, has_magnitudes)
        if key in self._compiled:
            return self._compiled[key]

        def evaluate(theta, context, magnitudes):
            return self._jax_engine(
                magnitudes,
                joint=joint,
                context=context,
            ).log_prob(theta, context=context)

        if has_context and has_magnitudes:
            fn = lambda theta, context, magnitudes: evaluate(
                theta, context, magnitudes
            )
        elif has_context:
            fn = lambda theta, context: evaluate(theta, context, None)
        elif has_magnitudes:
            fn = lambda theta, magnitudes: evaluate(theta, None, magnitudes)
        else:
            fn = lambda theta: evaluate(theta, None, None)
        self._compiled[key] = jax.jit(fn)
        return self._compiled[key]

    def log_density_batch(self, theta, *, context=None, magnitudes=None, joint=False):
        if joint and magnitudes is None:
            raise ValueError("joint source photometry requires magnitudes")
        evaluator = self.log_joint_density if joint else self.log_density
        if context is None and magnitudes is None:
            return jax.vmap(evaluator)(jnp.asarray(theta))
        return jax.vmap(
            lambda row: evaluator(row, context=context, magnitudes=magnitudes)
        )(jnp.asarray(theta))

    def to_physical(self, theta, *, context=None):
        return self._jax_engine().to_physical(theta, context=context)

    def to_deterministic_physical(self, theta, *, context=None):
        return self._jax_engine().to_deterministic_physical(theta, context=context)

    def to_derived(self, theta, *, context=None):
        if not hasattr(self.param_type, "to_derived"):
            return {}
        return dict(self.param_type.to_derived(theta, context))

    def log_abs_det_jacobian(self, theta, *, context=None):
        return self._jax_engine().log_abs_det_jacobian(theta, context=context)

    def sample_physical(self, theta, *, context=None, magnitudes=None, joint=False, rng=None):
        view = self._view(magnitudes, joint=joint, context=context)
        density = view._prior.density
        if bool(getattr(self.param_type, "uses_theta_mu_physical", False)) and (
            view.proposal is not None
        ):
            return self._sample_qmc_physical(
                theta, view, context=context, rng=rng, theta_mu=True
            )
        if bool(getattr(self.param_type, "uses_mu_physical", False)) and (
            view.direction_phi is not None
        ):
            return self._sample_qmc_physical(
                theta, view, context=context, rng=rng, theta_mu=False
            )
        return self._numpy_engine(
            magnitudes,
            joint=joint,
            context=context,
        ).sample_physical(
            theta,
            context=context,
            rng=rng,
        )

    def _isochrone_conditional_terms(self, theta, *, magnitudes, context=None):
        """Return paired-QMC conditional terms and physical candidates.

        ``magnitudes`` is a pytree whose leaves have leading dimension
        ``integration_samples``. thetaS and every hidden physical dimension
        use the same QMC row, avoiding a Cartesian thetaS-by-distance grid.

        The terms represent the physical density conditional on the supplied
        magnitudes. They exclude the marginal CMD factor
        ``log p(magnitudes)``; callers wanting the joint target must add that
        factor or use :meth:`log_joint_density`.
        """

        self._importance_proposal()
        key = (
            "isochrone_conditional",
            context is not None,
            tuple(sorted(magnitudes)),
        )
        if key not in self._compiled:
            if context is None:
                fn = lambda theta, magnitudes: self._isochrone_conditional_terms_impl(
                    theta, magnitudes, None
                )
            else:
                fn = lambda theta, context, magnitudes: self._isochrone_conditional_terms_impl(
                    theta, magnitudes, context
                )
            self._compiled[key] = jax.jit(fn)
        args = [jnp.asarray(theta)]
        if context is not None:
            args.append(context)
        args.append(magnitudes)
        return self._compiled[key](*args)

    def _isochrone_conditional_terms_impl(self, theta, magnitudes, context):
        proposal = self._importance_proposal()
        first_magnitude = next(iter(magnitudes.values()))
        if jnp.ndim(first_magnitude) == 0:
            one_proposal = self.galaxy._theta_star_proposal(
                magnitudes=magnitudes,
                context=context,
            )
            theta_proposal = type(one_proposal)(
                log_center=jnp.full_like(proposal.theta_u, one_proposal.log_center),
                log_sigma=jnp.full_like(proposal.theta_u, one_proposal.log_sigma),
            )
            magnitudes = {
                key: jnp.full_like(proposal.theta_u, value)
                for key, value in magnitudes.items()
            }
        else:
            theta_proposal = jax.vmap(
                lambda current: self.galaxy._theta_star_proposal(
                    magnitudes=current,
                    context=context,
                )
            )(magnitudes)
        log_sigma = theta_proposal.log_sigma
        z = ndtri(jnp.clip(proposal.theta_u, 1.0e-7, 1.0 - 1.0e-7))
        log_theta_s = theta_proposal.log_center + log_sigma * z
        theta_s = jnp.exp(log_theta_s)
        log_q_theta = (
            -0.5 * z**2
            - jnp.log(log_sigma)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )

        def current_context(value):
            result = {} if context is None else dict(context)
            result["thS"] = value
            return result

        def value_tuple(values):
            if isinstance(values, tuple):
                return values
            return tuple(values[:, index] for index in range(values.shape[1]))

        def conditional_log_density(values, current_theta_s, current_magnitudes):
            ml, dl, ds, mu_n, mu_e = values[:5]
            current = current_context(current_theta_s)
            value = self.galaxy.log_joint_density(
                (ml, dl, ds, mu_n, mu_e),
                magnitudes=current_magnitudes,
                theta_star_mas=current_theta_s,
                context=current,
            )
            reference_magnitude, color = self.galaxy.isochrone.values_from_magnitudes(
                current_magnitudes
            )
            value = value - self.galaxy._conditional_prior.source_prior.log_marginal_density(
                reference_magnitude,
                color,
                context=current,
            )
            return value + _physical_prior_sum(
                tuple(self._physical_priors),
                ML=ml,
                DL=dl,
                DS=ds,
                mu_N=mu_n,
                mu_E=mu_e,
                mu=jnp.hypot(mu_n, mu_e),
            )

        uses_theta_mu = bool(
            getattr(self.param_type, "uses_theta_mu_physical", False)
        )
        marginalizes_distance = bool(
            getattr(self.param_type, "marginalizes_distance", False)
        )
        uses_mu = bool(getattr(self.param_type, "uses_mu_physical", False))

        if uses_theta_mu:
            transformed = jax.vmap(
                lambda current_theta_s: self.param_type.to_theta_mu_physical(
                    theta, current_context(current_theta_s)
                )
            )(theta_s)
            theta_e, mu = transformed
            reduced_jacobian = jax.vmap(
                lambda current_theta_s: self.param_type.log_abs_det_jacobian(
                    theta, current_context(current_theta_s)
                )
            )(theta_s)
            ml, dl, ds, physical_jacobian = _mass_proposal_geometry(
                proposal, theta_e, mu
            )
            mu_n = mu * jnp.cos(proposal.phi)
            mu_e = mu * jnp.sin(proposal.phi)
            values = (ml, dl, ds, mu_n, mu_e)
            log_q_physical = (
                proposal.log_q_ds
                + proposal.log_q_mass
                - jnp.log(2.0 * jnp.pi)
            )
        elif marginalizes_distance:
            full_impl = self.param_type.distance_impl

            def transform(current_theta_s, ds):
                current = current_context(current_theta_s)
                full_theta = self.param_type.with_distance(theta, ds)
                return (
                    full_impl.to_physical(full_theta, current),
                    full_impl.log_abs_det_jacobian(full_theta, current),
                )

            values, reduced_jacobian = jax.vmap(transform)(theta_s, proposal.ds)
            values = value_tuple(values)
            physical_jacobian = 0.0
            log_q_physical = proposal.log_q_ds
        elif uses_mu:
            transformed = jax.vmap(
                lambda current_theta_s: self.param_type.to_mu_physical(
                    theta, current_context(current_theta_s)
                )
            )(theta_s)
            ml, dl, ds, mu = transformed
            reduced_jacobian = jax.vmap(
                lambda current_theta_s: self.param_type.log_abs_det_jacobian(
                    theta, current_context(current_theta_s)
                )
            )(theta_s)
            mu_n = mu * jnp.cos(proposal.phi)
            mu_e = mu * jnp.sin(proposal.phi)
            values = (ml, dl, ds, mu_n, mu_e)
            physical_jacobian = jnp.log(jnp.abs(mu)) + jnp.log(2.0 * jnp.pi)
            log_q_physical = jnp.zeros_like(theta_s)
        else:
            def transform(current_theta_s):
                current = current_context(current_theta_s)
                return (
                    self.param_type.to_physical(theta, current),
                    self.param_type.log_abs_det_jacobian(theta, current),
                )

            values, reduced_jacobian = jax.vmap(transform)(theta_s)
            values = value_tuple(values)
            physical_jacobian = 0.0
            log_q_physical = jnp.zeros_like(theta_s)

        logp = jax.vmap(conditional_log_density)(
            jnp.stack(values[:5], axis=1), theta_s, magnitudes
        )
        terms = (
            logp
            + reduced_jacobian
            + physical_jacobian
            - log_q_physical
            - log_q_theta
        )
        valid = jnp.isfinite(terms)
        terms = jnp.where(valid, terms, -jnp.inf)
        keys = ["ML", "DL", "DS", "mu_N", "mu_E"]
        keys.extend(getattr(self.param_type, "derived_names", ()))
        physical = {key: value for key, value in zip(keys, values)}
        physical["thetaS"] = theta_s
        return {"log_terms": terms, "physical": physical}

    def _sample_qmc_physical(self, theta, view, *, context, rng, theta_mu):
        rng = np.random.default_rng() if rng is None else rng
        array = np.asarray(theta)
        if array.ndim == 2:
            draws = [
                self._sample_qmc_physical(
                    row,
                    view,
                    context=context,
                    rng=rng,
                    theta_mu=theta_mu,
                )
                for row in array
            ]
            return {
                key: np.asarray([draw[key] for draw in draws])
                for key in draws[0]
            }
        engine = self._jax_engine(context=context)
        if theta_mu:
            theta_e, mu = engine.to_theta_mu_physical(theta, context=context)
            terms, values = view.theta_mu_log_terms(theta_e, mu)
            index = _draw_log_weight_index(terms, rng)
            ml, dl, ds, mu_n, mu_e = (float(value[index]) for value in values)
        else:
            ml, dl, ds, mu = engine.to_mu_physical(theta, context=context)
            terms, values = view.direction_log_terms(ml, dl, ds, mu)
            index = _draw_log_weight_index(terms, rng)
            mu_n, mu_e = (float(value[index]) for value in values)
            ml, dl, ds = float(ml), float(dl), float(ds)
        return {
            "ML": ml,
            "DL": dl,
            "DS": ds,
            "mu_N": mu_n,
            "mu_E": mu_e,
            "mu": float(np.hypot(mu_n, mu_e)),
        }


__all__ = ["ParameterizedGalaxyModel"]


def _build_integration_proposal(galaxy, samples, seed, *, with_mass):
    density = galaxy.density
    distance = np.asarray(density.distance.distance_pc, dtype=float) / 1000.0
    source = np.sum(
        np.asarray(density.distance.source_by_component, dtype=float), axis=1
    )
    source = np.maximum(source, 0.0)
    norm = float(np.trapezoid(source, distance))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("source-distance proposal has zero probability")
    source_density = source / norm
    segment_mass = 0.5 * (source_density[1:] + source_density[:-1]) * np.diff(distance)
    cdf = np.concatenate(([0.0], np.cumsum(segment_mass)))
    cdf /= cdf[-1]

    points = _halton_points(samples, 6 if with_mass else 5, seed)
    ds = np.interp(points[:, 0], cdf, distance)
    q_ds = np.interp(ds, distance, source_density)
    common = {
        "ds": jnp.asarray(ds),
        "phi": jnp.asarray(2.0 * np.pi * points[:, 3]),
        "log_q_ds": jnp.asarray(np.log(q_ds)),
        "theta_u": jnp.asarray(points[:, 4]),
    }

    if not with_mass:
        return _IntegrationProposal(**common)

    mass_density, mass_edges, broad_center, broad_sigma = _flow_mass_proposal(
        density, seed
    )
    mass_probability = mass_density * np.diff(mass_edges)
    mass_cdf = np.concatenate(([0.0], np.cumsum(mass_probability)))
    mass_cdf /= mass_cdf[-1]
    fitted_log_mass = np.interp(points[:, 2], mass_cdf, mass_edges)
    broad_log_mass = broad_center + broad_sigma * np.asarray(
        ndtri(jnp.asarray(np.clip(points[:, 2], 1.0e-7, 1.0 - 1.0e-7)))
    )
    fitted_fraction = 0.9
    log_mass = np.where(
        points[:, 1] < fitted_fraction,
        fitted_log_mass,
        broad_log_mass,
    )
    bin_index = np.clip(
        np.searchsorted(mass_edges, log_mass, side="right") - 1,
        0,
        len(mass_density) - 1,
    )
    inside_histogram = (log_mass >= mass_edges[0]) & (log_mass <= mass_edges[-1])
    q_log_mass_histogram = np.where(
        inside_histogram, mass_density[bin_index], 0.0
    )
    standardized_mass = (log_mass - broad_center) / broad_sigma
    q_log_mass_broad = (
        np.exp(-0.5 * standardized_mass**2)
        / (np.sqrt(2.0 * np.pi) * broad_sigma)
    )
    q_log_mass = (
        fitted_fraction * q_log_mass_histogram
        + (1.0 - fitted_fraction) * q_log_mass_broad
    )
    mass = np.exp(log_mass)
    q_mass = q_log_mass / mass
    component_density = np.column_stack([
        np.interp(
            ds,
            distance,
            np.asarray(density.distance.source_by_component)[:, component],
        )
        for component in range(
            np.asarray(density.distance.source_by_component).shape[1]
        )
    ])
    from gapmoe.density.flow_backend import SOURCE_GROUP_MATRIX_NP

    group_density = component_density @ SOURCE_GROUP_MATRIX_NP.T
    group_probability = group_density / np.sum(
        group_density, axis=1, keepdims=True
    )
    group_cdf = np.cumsum(group_probability, axis=1)
    source_group = np.minimum(
        np.sum(points[:, 5, None] > group_cdf, axis=1),
        group_probability.shape[1] - 1,
    )
    q_source_group = group_probability[
        np.arange(samples), source_group
    ]
    return _MassImportanceProposal(
        **common,
        mass=jnp.asarray(mass),
        log_q_mass=jnp.asarray(np.log(q_mass)),
        source_group=jnp.asarray(source_group),
        log_q_source_group=jnp.asarray(np.log(q_source_group)),
    )


def _flow_mass_proposal(density, seed, *, proposal_samples=16384, bins=96):
    sampler = getattr(density, "sample_source_group", None)
    kernel_sampler = getattr(density, "_sample_kernel", None)
    if sampler is None or kernel_sampler is None:
        raise TypeError("mass proposal requires a physical Flow sampler")

    source_key, kernel_key = jax.random.split(jax.random.key(seed))
    source_keys = jax.random.split(source_key, proposal_samples)
    kernel_keys = jax.random.split(kernel_key, proposal_samples)
    ds, group = jax.vmap(sampler)(source_keys)
    physical = jax.vmap(kernel_sampler)(kernel_keys, ds, group)
    mass = np.asarray(physical[:, 0], dtype=float)
    log_mass = np.log(mass[np.isfinite(mass) & (mass > 0.0)])
    if not len(log_mass):
        edges = np.linspace(np.log(1.0e-4), np.log(1.0e3), bins + 1)
        histogram = np.full(bins, 1.0 / (edges[-1] - edges[0]))
        return histogram, edges, np.log(0.3), 4.0

    broad_center = float(np.median(log_mass))
    broad_sigma = max(3.0, 3.0 * float(np.std(log_mass)))
    lower, upper = np.quantile(log_mass, (0.001, 0.999))
    padding = max(0.5, 0.1 * float(upper - lower))
    edges = np.linspace(lower - padding, upper + padding, bins + 1)
    counts, _ = np.histogram(log_mass, bins=edges)
    density_values = counts.astype(float) + 0.5
    density_values /= np.sum(density_values * np.diff(edges))
    return density_values, edges, broad_center, broad_sigma


def _halton_points(samples, dimensions, seed):
    primes = (2, 3, 5, 7, 11, 13)
    if dimensions > len(primes):
        raise ValueError("Halton generator supports at most six dimensions")
    indices = np.arange(1 + seed * samples, 1 + (seed + 1) * samples)
    return np.column_stack(
        [_radical_inverse(indices, base) for base in primes[:dimensions]]
    )


def _radical_inverse(indices, base):
    values = np.zeros(len(indices), dtype=float)
    factor = 1.0 / base
    current = indices.copy()
    while np.any(current):
        values += factor * (current % base)
        current //= base
        factor /= base
    return values


def _draw_log_weight_index(log_weights, rng):
    values = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("conditional physical distribution has zero probability")
    peak = np.max(values[finite])
    weights = np.where(finite, np.exp(values - peak), 0.0)
    weights /= np.sum(weights)
    return int(rng.choice(len(values), p=weights))


def _physical_prior_sum(priors, **values):
    total = jnp.asarray(0.0)
    for fn in priors:
        total = total + fn(**_accepted_kwargs(fn, values))
    return total


def _accepted_kwargs(fn, values):
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return values
    accepts_all = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_all:
        return values
    accepted = {
        name: values[name]
        for name, parameter in signature.parameters.items()
        if name in values
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in accepted
    ]
    if missing:
        raise TypeError(
            "physical prior requested unavailable value(s): " + ", ".join(missing)
        )
    return accepted
