import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from jax import jit
from jax import vmap
from functools import partial

@jit
def physical_to_lightcurve_circular(theta,thS):
    tE     = jnp.exp(theta[1])
    rho    = jnp.exp(theta[3])
    s      = jnp.exp(theta[5])
    piEN   = theta[7]
    piEE   = theta[8]
    gamma1 = theta[9]
    gamma2 = theta[10]
    gamma3 = theta[11]

    piE = jnp.sqrt(piEE**2 + piEN**2)

    thE = thS / rho
    ML = thE / 8.1439 / piE # KAPPA = 8.1429 [mas / Msun]
    murel = thE / tE * 365.25
    murel_E = murel * piEE / piE
    murel_N = murel * piEN / piE

    gamma_sq = gamma1**2 + gamma2**2 + gamma3**2
    gamma_parallel = jnp.sqrt(gamma1**2 + gamma3**2)
    gamma_ratio = jnp.sqrt(1 + (gamma1/gamma3)**2)
    orbital_scale = jnp.cbrt((s**3) * gamma_sq * gamma_ratio / (ML * 2.959122082855911e-4)) # G = 2.959122082855911e-4 [AU^3 / (Msun * day^2)]
    gamma_abs = jnp.sqrt(gamma_sq)
    Ds = 1 / ((orbital_scale - piE) * thE)

    pi_rel = thE * piE
    pi_S = 1 / Ds
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L

    RE = DL * thE
    orbital_radi = RE * s * gamma_ratio

    cosi = gamma2 / (gamma_ratio * gamma_abs)
    tanphi = - gamma1 * gamma_abs / (gamma3 * gamma_parallel)

    return  jnp.array([murel_E, murel_N, ML, DL, Ds, orbital_radi, cosi, tanphi])

@jit
def wrapped_physical_to_lightcurve_circular(theta_reduced, thS):
    tE, rho, s, piEN, piEE, gamma1, gamma2, gamma3 = theta_reduced

    piE = jnp.sqrt(piEE**2 + piEN**2)
    thE = thS / rho
    ML = thE / 8.1439 / piE
    murel = thE / tE * 365.25
    murel_E = murel * piEE / piE
    murel_N = murel * piEN / piE

    gamma_sq = gamma1**2 + gamma2**2 + gamma3**2
    gamma_parallel = jnp.sqrt(gamma1**2 + gamma3**2)
    gamma_ratio = jnp.sqrt(1 + (gamma1/gamma3)**2)
    orbital_scale = jnp.cbrt((s**3) * gamma_sq * gamma_ratio / (ML * 2.959122082855911e-4))
    gamma_abs = jnp.sqrt(gamma_sq)
    Ds = 1 / ((orbital_scale - piE) * thE)

    pi_rel = thE * piE
    pi_S = 1 / Ds
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L

    RE = DL * thE
    orbital_radi = RE * s * gamma_ratio

    cosi = gamma2 / (gamma_ratio * gamma_abs)
    tanphi = - gamma1 * gamma_abs / (gamma3 * gamma_parallel)

    return jnp.array([murel_E, murel_N, ML, DL, Ds, orbital_radi, cosi, tanphi])

@jit
def calc_ln_det_jacobian_circular(theta, thS):
    theta_reduced = jnp.array([
        jnp.exp(theta[1]),
        jnp.exp(theta[3]),
        jnp.exp(theta[5]),
        theta[7],
        theta[8],
        theta[9],
        theta[10],
        theta[11]
    ])
    J = jacfwd(wrapped_physical_to_lightcurve_circular)(theta_reduced, thS)
    sign, lndet = jnp.linalg.slogdet(J)
    return lndet
