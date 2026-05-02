#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from jax import jit
from jax import vmap
import emcee
import h5py
import corner
import matplotlib.pyplot as plt
import corner
import VBMicrolensing
VBM = VBMicrolensing.VBMicrolensing()
import tqdm
import math
import sys
import multiprocessing
sys.path.append("/moao38_7/nunota/gapmoe/src/gapmoe/")
from gapmoe import gapmoe
import EarthMotion
import parametrics
JD0 = 2450000


args = sys.argv
for i in range(len(args)):
  if args[i]  == "-output":
    output_path = args[i+1]
  if args[i]  == "-save":
    backend_path = args[i+1]
  if args[i]  == "-seed":
    seed = int(args[i+1])
# In[2]:


tref = 10063.874
coords = "17:57:38.03 -28:38:28.53"

VBM.t0_par = tref+JD0
VBM.parallaxsystem = 1
VBM.SetObjectCoordinates(coords)

RA_str,Dec_str = "17:57:38.03","-28:38:28.53"
RA_deg = EarthMotion.hms_string_to_degrees(RA_str)
Dec_deg = EarthMotion.dms_string_to_degrees(Dec_str)
vEarth = EarthMotion.calc_vEarth(tref,RA_deg,Dec_deg)

gapmoe_model = gapmoe(RA_deg,Dec_deg)
gapmoe_model.set_data()


# In[3]:


def mag2flux(mag):
    return 10**(-mag / 2.5)

def flux2mag(flux):
    return -2.5 * np.log10(flux)

def linear_fit(x, y, w):
    w_sum = np.sum(w)
    wxy_sum = np.sum(w * x * y)
    wx_sum = np.sum(w * x)
    wy_sum = np.sum(w * y)
    wxx_sum = np.sum(w * x * x)

    bunbo = w_sum * wxx_sum - wx_sum ** 2

    if bunbo == 0 or not np.isfinite(bunbo):
        return np.nan, np.nan, np.inf

    a = (w_sum * wxy_sum - wx_sum * wy_sum) / bunbo
    b = (wxx_sum * wy_sum - wx_sum * wxy_sum) / bunbo

    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan, np.nan, np.inf

    y_fit = a * x + b
    chi2 = np.sum(w * (y - y_fit) ** 2)

    return a, b, chi2

cVIBoya = 0.50141358
dVIBoya = 0.41968496
EVI, AI = 1.483,  1.822
def calc_thS(_fs_I,_fs_v):
    I_S = flux2mag(_fs_I)
    V_S = flux2mag(_fs_v)
    VI_S = V_S - I_S
    
    I0_S = I_S - AI
    VI0_S = VI_S - EVI
    
    _theta_star  = 0.5*10**(cVIBoya + dVIBoya*VI0_S  - 0.2*I0_S)
    return _theta_star


# In[4]:


loaded = np.load("/moao38_7/nunota/gapmoe/simu_data/simu_data_kepler_01.npz", allow_pickle=True)

simu_data = {k: loaded[k] for k in loaded.files}

def calc_ln_like(theta,source_flux=False):
    _t0 = theta[0]
    _tE     = theta[1]
    _u0 = theta[2]
    _rho    = theta[3]
    _q = theta[4]
    _s      = theta[5]
    _alpha = theta[6]
    _piEN   = theta[7]
    _piEE   = theta[8]
    _gamma1 = theta[9]
    _gamma2 = theta[10]
    _gamma3 = theta[11]
    _rs = theta[12]
    _as = theta[13] 
    _pr = [math.log(_s), math.log(_q), _u0, _alpha, math.log(_rho), math.log(_tE), _t0+JD0, _piEN, _piEE, _gamma1,_gamma2,_gamma3,_rs,_as]

    _chi2_sum = 0
    for _name in simu_data.keys():
        _model_amp = np.array(VBM.BinaryLightCurveKepler(_pr,simu_data[_name]["time"]+JD0)[0])
        _fs,_fb,_chi2 = linear_fit(_model_amp,simu_data[_name]["flux_obs"],simu_data[_name]["ferr"]**(-2))
        _chi2_sum += _chi2
        if _name == 'MOA_Red':
            _fs_i = _fs
        if _name == 'MOA_V':
            _fs_v = _fs

    if source_flux:
        return -0.5*_chi2_sum, _fs_i, _fs_v
    else:
        return -0.5*_chi2_sum
    
def calc_ln_prior(theta, thS, vEarth):
    t0, u0, q, ML, DL, DS, murel_N_hel, murel_E_hel,orbital_radi, e, cos_i, Om_NE, om, nu = parametrics.lightcurve_to_physical_kepler(theta, thS, vEarth)
    
    mu_abs = jnp.sqrt(murel_E_hel**2 + murel_N_hel**2)
    mu_phi = jnp.arctan2(murel_E_hel, murel_N_hel)
    DL *= 1e3
    DS *= 1e3

    if not (10000 <= t0 <= 11000):
        return -np.inf
    if not (-5 <= u0 <= 5):
        return -np.inf
    if not (1e-5 <= q <= 1.0):
        return -np.inf
    if not (0 <= e <= 1.0):
        return -np.inf
    if not (-np.pi <= Om_NE <= np.pi):
        return -np.inf
    if not (-np.pi <= om <= np.pi):
        return -np.inf
    if not (-np.pi <= nu <= np.pi):
        return -np.inf
    if not (0 <= orbital_radi <= 1e4):
        return -np.inf

    if not (gapmoe_model.M_MIN <= ML <= gapmoe_model.M_MAX):
        return -np.inf
    if not (gapmoe_model.DL_MIN <= DL < gapmoe_model.DL_MAX):
        return -np.inf
    if not (gapmoe_model.DS_MIN <= DS < gapmoe_model.DS_MAX):
        return -np.inf
    if DL >= DS:
        return -np.inf
    if not (gapmoe_model.MU_MIN <= mu_abs < gapmoe_model.MU_MAX):
        return -np.inf

    return gapmoe_model.log_galactic_prior(ML, DL, DS, mu_abs, mu_phi) - np.log(mu_abs) #jacobian for (muN, muE) -> (mu,phi)

def check_positive(theta):
    _t0 = theta[0]
    _tE     = theta[1]
    _u0 = theta[2]
    _rho    = theta[3]
    _q = theta[4]
    _s      = theta[5]
    _alpha = theta[6]
    _piEN   = theta[7]
    _piEE   = theta[8]
    _gamma1 = theta[9]
    _gamma2 = theta[10]
    _gamma3 = theta[11]
    _rs = theta[12]
    _as = theta[13] 
    
    return _s > 0 and _q > 0 and _rho > 0 and _tE > 0


def check_eccentricity(theta):
    s      = theta[5]
    gamma1 = theta[9]
    gamma2 = theta[10]
    gamma3 = theta[11]
    r_s = theta[12]
    a_s = theta[13] + 1e-8 

    GM_by_RE3_s3 = a_s * np.sqrt(1+r_s**2) * (gamma1**2 + gamma2**2 + gamma3**2)/(2*a_s - 1)

    r = jnp.array([1,0, r_s])
    v = jnp.array([gamma1,gamma2,gamma3])
    
    h = jnp.cross(r,v)
    A = jnp.cross(v,h) / GM_by_RE3_s3  - r / jnp.sqrt(jnp.dot(r,r))
    e = jnp.sqrt(jnp.dot(A,A))

    return e < 0.99

def ln_uniform_prior(theta, thS, vEarth):
    t0, u0, q, ML, DL, DS, murel_N_hel, murel_E_hel,orbital_radi, e, cos_i, Om_NE, om, nu = parametrics.lightcurve_to_physical_kepler(theta, thS, vEarth)
    return -np.log(q) - np.log(orbital_radi)
    
def ln_prob(theta):
    if not check_positive(theta):
        return -np.inf, np.nan

    if not check_eccentricity(theta):
        return -np.inf, np.nan  

    _ln_like, _fs_i, _fs_v = calc_ln_like(theta,source_flux=True)
    if np.isnan(_fs_i) or np.isnan(_fs_v) or np.isneginf(_ln_like):
        return -np.inf, np.nan

    _thS = calc_thS(_fs_i,_fs_v)
    _ln_det = parametrics.calc_ln_det_jacobian_kepler(theta,_thS,vEarth)
    _ln_prior = calc_ln_prior(theta,_thS,vEarth)
    _ln_uni_prior = ln_uniform_prior(theta,_thS,vEarth)
        
    if np.isneginf(_ln_det) or np.isneginf(_ln_prior) or np.isneginf(_ln_uni_prior):
        return -np.inf, np.nan

    ln_post = _ln_like + _ln_prior + _ln_det + _ln_uni_prior
    return ln_post, _thS

backend_path = "/home/nunota/work/gapmoe/test_result/backend/rerun_simu_01_kepler_chain_01.h5"

new_max_steps = 200000
autocorr_check_interval = 1000

reader = emcee.backends.HDFBackend(backend_path, read_only=True)
chain = reader.get_chain()

p0 = chain[-1]

nwalkers, ndim = p0.shape

backend = emcee.backends.HDFBackend("/home/nunota/work/gapmoe/test_result/backend/rerun_simu_01_kepler_chain_03.h5")
sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, backend=backend)

old_tau = np.inf

for sample in sampler.sample(p0, iterations=new_max_steps, progress=True):
    if sampler.iteration % autocorr_check_interval == 0:
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except emcee.autocorr.AutocorrError:
            tau = np.ones(ndim) * np.nan

        acceptance_rate = np.mean(sampler.acceptance_fraction)
        flat_samples = sampler.get_chain(flat=True)
        flat_log_probs = sampler.get_log_prob(flat=True)
        best_idx = np.argmax(flat_log_probs)
        current_best = flat_samples[best_idx]

        with open(output_path, "a") as f:
            print(f"[manual-resume] step {sampler.iteration}: autocorr time = {tau}", file=f)
            print(f"[manual-resume] acceptance rate = {acceptance_rate:.3f}", file=f)
            print(f"[manual-resume] current best params = {current_best}", file=f)

        if not np.any(np.isnan(tau)):
            converged = np.all(tau * 100 < sampler.iteration)
            stable = np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged and stable:
                with open(output_path, "a") as f:
                    print(f"[manual-resume] converged at step {sampler.iteration}", file=f)
                break
            old_tau = tau
