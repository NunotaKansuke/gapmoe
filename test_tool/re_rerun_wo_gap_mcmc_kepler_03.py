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

loaded = np.load("/moao38_7/nunota/gapmoe/simu_data/simu_data_03_kepler.npz", allow_pickle=True)

simu_data = {k: loaded[k] for k in loaded.files}
for _name in simu_data.keys():
    simu_data[_name]["ferr"] /= np.sqrt(10)

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
    
def calc_ln_prior(theta):
    _t0     = theta[0]
    _tE     = theta[1]
    _u0     = theta[2]
    _rho    = theta[3]
    _q      = theta[4]
    _s      = theta[5]
    _alpha  = theta[6]
    _piEN   = theta[7]
    _piEE   = theta[8]
    _gamma1 = theta[9]
    _gamma2 = theta[10]
    _gamma3 = theta[11]
    _rs     = theta[12]
    _as     = theta[13]

    if not (10000 <= _t0 <= 11000): return -np.inf
    if not (0 < _tE <= 1000): return -np.inf
    if not (-5 <= _u0 <= 5): return -np.inf
    if not (0 < _rho <= 1): return -np.inf
    if not (1e-5 <= _q <= 1): return -np.inf
    if not (0 < _s <= 5): return -np.inf
    if not (-np.pi <= _alpha <= np.pi): return -np.inf
    if not (-3 <= _piEN <= 3): return -np.inf
    if not (-3 <= _piEE <= 3): return -np.inf
    if not (-1 <= _gamma1 <= 1): return -np.inf
    if not (-1 <= _gamma2 <= 1): return -np.inf
    if not (-1 <= _gamma3 <= 1): return -np.inf
    if not (-10 <= _rs <= 10): return -np.inf
    if not (1 <= _as <= 10): return -np.inf

    return -np.log(_q)

def ln_prob(theta):
    _ln_prior = calc_ln_prior(theta)
        
    if jnp.isneginf(_ln_prior):
        return -jnp.inf, np.nan

    if not check_eccentricity(theta):
        return -np.inf, np.nan  

    _ln_like, _fs_i, _fs_v = calc_ln_like(theta,source_flux=True)
    if jnp.isnan(_fs_i) or jnp.isnan(_fs_v) or jnp.isneginf(_ln_like):
        return -jnp.inf, np.nan

    _thS = calc_thS(_fs_i,_fs_v)

    ln_post = _ln_like + _ln_prior
    return ln_post, _thS

def read_chain(path,burnin,thin,flat=True):
    sampler = emcee.backends.HDFBackend(path)
    chain = sampler.get_chain(flat=flat, discard=burnin, thin=thin)
    blob = sampler.get_blobs(flat=flat, discard=burnin, thin=thin) 
    lnprob = sampler.get_log_prob(flat=flat, discard=burnin, thin=thin) 
    return chain, blob,lnprob

#backend_path = "/home/nunota/work/gapmoe/test_result/backend/rerun_wo_gap_simu_02_kepler_chain_01.h5"
#reader = emcee.backends.HDFBackend(backend_path, read_only=True)
#chain = reader.get_chain()
#p0 = chain[-1]

kep_wo_chain_03,kep_wo_blob_03,kep_wo_prob_03 = read_chain("/moao38_7/nunota/gapmoe/test_result/rogue1/backend/rerun_wo_gap_simu_03_kepler_chain_01.h5",0,1,flat=False)
use_ind_wo_03 = np.where((np.abs(kep_wo_chain_03[-1,:,1]) > 73))[0]
samples = kep_wo_chain_03[100000:, use_ind_wo_03, :].reshape(-1, kep_wo_chain_03.shape[-1])
p0 = samples[np.random.choice(len(samples), size=28, replace=False)]

new_max_steps = 300000
autocorr_check_interval = 1000

nwalkers, ndim = p0.shape

backend = emcee.backends.HDFBackend("/home/nunota/work/gapmoe/test_result/backend/rerun_wo_gap_simu_03_kepler_chain_02.h5")
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
