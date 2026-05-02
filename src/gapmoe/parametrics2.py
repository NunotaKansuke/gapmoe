import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from jax import jit
from jax import vmap
from functools import partial

@jit
def lightcurve_to_physical_kepler(theta,thS,vEarth, G = 2.959122082855911e-4, KAPPA = 8.1429):
    # G [AU^3 / (Msun * day^2)] , KAPPA [mas / Msun]
    t0 = theta[0]
    tE     = theta[1]
    u0 = theta[2]
    rho    = theta[3]
    q = theta[4]
    s      = theta[5]
    alpha = theta[6]
    piEN   = theta[7]
    piEE   = theta[8]
    gamma1 = theta[9]
    gamma2 = theta[10]
    gamma3 = theta[11]
    sz = theta[12]
    Ds = theta[13] 
    vEarth_N, vEarthE = vEarth
    
    G = 2.959122082855911e-4 # [AU^3 / (Msun * day^2)]
    KAPPA = 8.1429 # [mas / Msun]

    piE = jnp.sqrt(piEN**2 + piEE**2)

    thE = thS / rho #mas
    ML = thE / KAPPA / piE #Msun
    murel_geo = thE / tE * 365.25 # mas / year
    murel_N_geo = murel_geo * piEN / piE # mas / year
    murel_E_geo = murel_geo * piEE / piE # mas / year

    pi_rel = thE * piE
    pi_S = 1 / DS
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L #kpc
    RE = thE * DL #AU
    r = RE* jnp.array([s,0,s_z])
    v = RE* jnp.array([s*gamma1,gamma2,s*gamma3])
    eps = jnp.dot(v,v)/2 - mu/jnp.sqrt(jnp.dot(r,r))
    a = -mu/(2*eps)

    murel_N_hel = murel_N_geo + thE * piE * vEarth[0]
    murel_E_hel = murel_E_geo + thE * piE * vEarth[1]

    RE = DL * thE # AU
    orbital_radi = RE * a_norm #AU

    r = RE * s * jnp.array([1,0, r_s]) #AU
    v = RE * s * jnp.array([gamma1,gamma2,gamma3]) #AU / day
    
    h = jnp.cross(r,v)
    A = jnp.cross(v,h) / (G*ML) - r / jnp.sqrt(jnp.dot(r,r))
    e = jnp.sqrt(jnp.dot(A,A))
    
    z = h / jnp.sqrt(jnp.dot(h,h))
    x = A / e
    y = jnp.cross(z,x)
    
    cos_i = z[2]
    sin_i = jnp.sqrt(1 - cos_i**2)
    
    sin_Om0, cos_Om0 = z[0] / sin_i, - z[1] / sin_i
    Om0 = jnp.arctan2(sin_Om0, cos_Om0)
    Om_NE = Om0 + jnp.arctan2(piEE,piEN) - alpha
    Om_NE = jnp.arctan2(jnp.sin(Om_NE),jnp.cos(Om_NE))
    
    sin_om, cos_om = x[2] / sin_i, y[2] / sin_i
    om = jnp.arctan2(sin_om, cos_om)
    
    cos_nu = jnp.dot(r,x)/jnp.sqrt(jnp.dot(r,r))
    sin_nu = jnp.dot(r,y)/jnp.sqrt(jnp.dot(r,r))
    nu = jnp.arctan2(sin_nu,cos_nu)

    return  jnp.array([t0, u0, q, ML, DL, DS, murel_N_hel, murel_E_hel, orbital_radi, e, cos_i, Om_NE, om, nu])

@jit
def calc_ln_det_jacobian_kepler(theta, thS,vEarth):
    J = jacfwd(lightcurve_to_physical_kepler)(theta, thS,vEarth)
    sign, lndet = jnp.linalg.slogdet(J)
    return lndet

