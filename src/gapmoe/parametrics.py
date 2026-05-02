import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from jax import jit
from jax import vmap
from functools import partial

@jit
def lightcurve_to_physical_circular(theta,thS,vEarth, G = 2.959122082855911e-4, KAPPA = 8.1429):
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
    vEarth_N, vEarthE = vEarth
    
    G = 2.959122082855911e-4 # [AU^3 / (Msun * day^2)]
    KAPPA = 8.1429 # [mas / Msun]

    piE = jnp.sqrt(piEN**2 + piEE**2)

    thE = thS / rho #mas
    ML = thE / KAPPA / piE #Msun
    murel_geo = thE / tE * 365.25 # mas / year
    murel_N_geo = murel_geo * piEN / piE # mas / year
    murel_E_geo = murel_geo * piEE / piE # mas / year

    gamma_sq = gamma1**2 + gamma2**2 + gamma3**2
    gamma_parallel = jnp.sqrt(gamma1**2 + gamma3**2)
    gamma_ratio = jnp.sqrt(1 + (gamma1/gamma3)**2)
    orbital_scale = jnp.cbrt((s**3) * gamma_sq * gamma_ratio / (ML * G))
    gamma_abs = jnp.sqrt(gamma_sq)
    DS = 1 / ((orbital_scale - piE) * thE) #kpc

    pi_rel = thE * piE
    pi_S = 1 / DS
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L #kpc
    
    murel_N_hel = murel_N_geo + thE * piE * vEarth[0]
    murel_E_hel = murel_E_geo + thE * piE * vEarth[1]
    
    RE = DL * thE # AU
    orbital_radi = RE * s * gamma_ratio #AU

    r = RE * s * jnp.array([1,0, - gamma1 / gamma3]) #AU
    v = RE * s * jnp.array([gamma1,gamma2,gamma3]) #AU / day
    
    h = jnp.cross(r,v)
    
    z = h / jnp.sqrt(jnp.dot(h,h))
    
    cos_i = z[2]
    sin_i = jnp.sqrt(1 - cos_i**2)
    sin_Om0, cos_Om0 = z[0] / sin_i, - z[1] / sin_i
    Om0 = jnp.arctan2(sin_Om0, cos_Om0)
    Om_NE = Om0 + jnp.arctan2(piEE,piEN) - alpha
    Om_NE = jnp.arctan2(jnp.sin(Om_NE),jnp.cos(Om_NE))
    
    x = jnp.array([cos_Om0,sin_Om0,0])
    y = jnp.cross(z,x)
    
    cos_phi0 = jnp.dot(r,x)/jnp.sqrt(jnp.dot(r,r))
    sin_phi0 = jnp.dot(r,y)/jnp.sqrt(jnp.dot(r,r))
    phi0 = jnp.arctan2(sin_phi0,cos_phi0)

    return  jnp.array([t0, u0, q, ML, DL, DS, murel_N_hel, murel_E_hel, orbital_radi, cos_i, Om_NE, phi0])

@jit
def calc_ln_det_jacobian_circular(theta, thS,vEarth):
    J = jacfwd(lightcurve_to_physical_circular)(theta, thS,vEarth)
    sign, lndet = jnp.linalg.slogdet(J)
    return lndet

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
    r_s = theta[12]
    a_s = theta[13] 
    vEarth_N, vEarthE = vEarth
    
    G = 2.959122082855911e-4 # [AU^3 / (Msun * day^2)]
    KAPPA = 8.1429 # [mas / Msun]

    piE = jnp.sqrt(piEN**2 + piEE**2)

    thE = thS / rho #mas
    ML = thE / KAPPA / piE #Msun
    murel_geo = thE / tE * 365.25 # mas / year
    murel_N_geo = murel_geo * piEN / piE # mas / year
    murel_E_geo = murel_geo * piEE / piE # mas / year


    gamma_sq = gamma1**2 + gamma2**2 + gamma3**2
    gamma_abs = jnp.sqrt(gamma_sq)
    a_norm = a_s * s * jnp.sqrt(1 + r_s**2)
    orbital_scale = jnp.cbrt((s**2) * a_norm * gamma_sq / (ML * G) / (2 * a_s - 1))
    orbital_scale = jnp.cbrt((s**3) * a_s * jnp.sqrt(1 + r_s**2) * gamma_sq / (ML * G) / (2 * a_s - 1))
    DS = 1 / ((orbital_scale - piE) * thE) #kpc

    pi_rel = thE * piE
    pi_S = 1 / DS
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L #kpc
    
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


@jit
def lightcurve_to_physical_circular_use_thE(theta, vEarth, G = 2.959122082855911e-4, KAPPA = 8.1429):
    # G [AU^3 / (Msun * day^2)] , KAPPA [mas / Msun]
    t0 = theta[0]
    tE     = theta[1]
    u0 = theta[2]
    thE    = theta[3] #mas
    q = theta[4]
    s      = theta[5]
    alpha = theta[6]
    piEN   = theta[7]
    piEE   = theta[8]
    gamma1 = theta[9]
    gamma2 = theta[10]
    gamma3 = theta[11]
    vEarth_N, vEarthE = vEarth
    


    piE = jnp.sqrt(piEN**2 + piEE**2)

    ML = thE / KAPPA / piE #Msun
    murel_geo = thE / tE * 365.25 # mas / year
    murel_N_geo = murel_geo * piEN / piE # mas / year
    murel_E_geo = murel_geo * piEE / piE # mas / year

    gamma_sq = gamma1**2 + gamma2**2 + gamma3**2
    gamma_parallel = jnp.sqrt(gamma1**2 + gamma3**2)
    gamma_ratio = jnp.sqrt(1 + (gamma1/gamma3)**2)
    orbital_scale = jnp.cbrt((s**3) * gamma_sq * gamma_ratio / (ML * G))
    gamma_abs = jnp.sqrt(gamma_sq)
    DS = 1 / ((orbital_scale - piE) * thE) #kpc

    pi_rel = thE * piE
    pi_S = 1 / DS
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L #kpc
    
    murel_N_hel = murel_N_geo + thE * piE * vEarth[0]
    murel_E_hel = murel_E_geo + thE * piE * vEarth[1]
    
    RE = DL * thE # AU
    orbital_radi = RE * s * gamma_ratio #AU

    r = RE * s * jnp.array([1,0, - gamma1 / gamma3]) #AU
    v = RE * s * jnp.array([gamma1,gamma2,gamma3]) #AU / day
    
    h = jnp.cross(r,v)
    
    z = h / jnp.sqrt(jnp.dot(h,h))
    
    cos_i = z[2]
    sin_i = jnp.sqrt(1 - cos_i**2)
    sin_Om0, cos_Om0 = z[0] / sin_i, - z[1] / sin_i
    Om0 = jnp.arctan2(sin_Om0, cos_Om0)
    Om_NE = Om0 + jnp.arctan2(piEE,piEN) - alpha
    Om_NE = jnp.arctan2(jnp.sin(Om_NE),jnp.cos(Om_NE))
    
    x = jnp.array([cos_Om0,sin_Om0,0])
    y = jnp.cross(z,x)
    
    cos_phi0 = jnp.dot(r,x)/jnp.sqrt(jnp.dot(r,r))
    sin_phi0 = jnp.dot(r,y)/jnp.sqrt(jnp.dot(r,r))
    phi0 = jnp.arctan2(sin_phi0,cos_phi0)

    return  jnp.array([t0, u0, q, ML, DL, DS, murel_N_hel, murel_E_hel, orbital_radi, cos_i, Om_NE, phi0])

@jit
def calc_ln_det_jacobian_circular_use_thE(theta,vEarth):
    J = jacfwd(lightcurve_to_physical_circular_use_thE)(theta,vEarth)
    sign, lndet = jnp.linalg.slogdet(J)
    return lndet

@jit
def lightcurve_to_physical_single(theta,thS,vEarth, G = 2.959122082855911e-4, KAPPA = 8.1429):
    # G [AU^3 / (Msun * day^2)] , KAPPA [mas / Msun]
    t0 = theta[0]
    tE     = theta[1]
    u0 = theta[2]
    rho    = theta[3]
    piEN   = theta[4]
    piEE   = theta[5]
    DS = theta[6]
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
    
    murel_N_hel = murel_N_geo + thE * piE * vEarth[0]
    murel_E_hel = murel_E_geo + thE * piE * vEarth[1]

    return  jnp.array([t0, u0, ML, DL, DS, murel_N_hel, murel_E_hel])

@jit
def calc_ln_det_jacobian_single(theta, thS,vEarth):
    J = jacfwd(lightcurve_to_physical_single)(theta, thS,vEarth)
    sign, lndet = jnp.linalg.slogdet(J)
    return lndet

@jit
def lightcurve_to_physical_single_use_thE(theta, vEarth, G = 2.959122082855911e-4, KAPPA = 8.1429):
    # G [AU^3 / (Msun * day^2)] , KAPPA [mas / Msun]
    t0 = theta[0]
    tE     = theta[1]
    u0 = theta[2]
    thE    = theta[3] #mas
    piEN   = theta[4]
    piEE   = theta[5]
    DS = theta[6]
    vEarth_N, vEarthE = vEarth
    
    piE = jnp.sqrt(piEN**2 + piEE**2)

    ML = thE / KAPPA / piE #Msun
    murel_geo = thE / tE * 365.25 # mas / year
    murel_N_geo = murel_geo * piEN / piE # mas / year
    murel_E_geo = murel_geo * piEE / piE # mas / year

    pi_rel = thE * piE
    pi_S = 1 / DS
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L #kpc
    
    murel_N_hel = murel_N_geo + thE * piE * vEarth[0]
    murel_E_hel = murel_E_geo + thE * piE * vEarth[1]
    
    return  jnp.array([t0, u0, ML, DL, DS, murel_N_hel, murel_E_hel])

@jit
def calc_ln_det_jacobian_single_use_thE(theta,vEarth):
    J = jacfwd(lightcurve_to_physical_single_use_thE)(theta,vEarth)
    sign, lndet = jnp.linalg.slogdet(J)
    return lndet