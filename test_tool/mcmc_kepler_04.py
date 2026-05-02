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
JD0 = 0


args = sys.argv
for i in range(len(args)):
  if args[i]  == "-output":
    output_path = args[i+1]
  if args[i]  == "-save":
    backend_path = args[i+1]
  if args[i]  == "-seed":
    seed = int(args[i+1])
# In[2]:


tref = 10090
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


# In[5]:


loaded = np.load("/moao38_7/nunota/gapmoe/simu_data/simu_data_04_kepler.npz", allow_pickle=True)

simu_data = {k: loaded[k] for k in loaded.files}

Is = 20 + AI
Vs = 1 + 20 + EVI
fs_I = mag2flux(Is)
fs_V = mag2flux(Vs)

t0_true = 10090
q_true = 0.01
u0_true = -0.02
alpha_true = -0.347735
tE_true = 151.64
rho_true = 0.00177562
s_true = 0.903443
piEN_true = 0.30033
piEE_true = -0.0714238
gamma1_true = -0.00125154
gamma2_true = 0.000376098
gamma3_true = -0.00356426
r_s_true = -0.141818
a_s_true = 0.954969
init_params = np.array([t0_true,tE_true,u0_true,rho_true,q_true,s_true,alpha_true,piEN_true,piEE_true,gamma1_true,gamma2_true,gamma3_true,r_s_true,a_s_true])


# In[6]:


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

    return gapmoe_model.log_galactic_prior(ML, DL, DS, mu_abs, mu_phi) - np.log(mu_abs)

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

def ln_uniform_prior(theta, thS, vEarth):
    t0, u0, q, ML, DL, DS, murel_N_hel, murel_E_hel,orbital_radi, e, cos_i, Om_NE, om, nu = parametrics.lightcurve_to_physical_kepler(theta, thS, vEarth)
    return -np.log(q) - np.log(orbital_radi)
    
def ln_prob(theta):
    if not check_positive(theta):
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

def read_chain(path,burnin,thin,flat=True):
    sampler = emcee.backends.HDFBackend(path)
    chain = sampler.get_chain(flat=flat, discard=burnin, thin=thin)
    blob = sampler.get_blobs(flat=flat, discard=burnin, thin=thin)
    lnprob = sampler.get_log_prob(flat=flat, discard=burnin, thin=thin)
    return chain, blob,lnprob
# In[7]:


param_stds = [
    0.01,     # t0 [days]
    0.1,      # tE
    0.005,    # u0
    3e-4,     # rho
    0.0001,    # q
    0.01,     # s
    0.05,     # alpha [rad]
    0.00001,     # piEN
    0.00001,     # piEE
    3e-4,     # gamma1
    3e-4,     # gamma2
    3e-4,      # gamma3
    1e-2,
    1e-2
]

ndim = len(init_params)
nwalkers = 28
max_nsteps = 300000
autocorr_check_interval = 100 

np.random.seed(45)
#p0 = np.array([
#    np.array(init_params) + np.random.normal(scale=param_stds)
#    for _ in range(nwalkers)
#])

kep_gap_chain_04,kep_gap_blob_04,kep_gap_prob_04 = read_chain("/moao38_7/nunota/gapmoe/test_result/rogue1/backend/rerun_simu_04_kepler_01.h5",0,1,flat=False)
samples = kep_gap_chain_04[5000:,np.where(kep_gap_prob_04[-1]>-10000)[0],:].reshape(-1, kep_gap_chain_04.shape[-1])[::1000]
p0 = samples[np.random.choice(len(samples), size=28, replace=False)] 
p0 += 1e-3 * np.std(samples, axis=0) * np.random.randn(*p0.shape)

np.random.seed(seed)

backend = emcee.backends.HDFBackend(backend_path)
backend.reset(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob,backend=backend)

old_tau = np.inf
for sample in sampler.sample(p0, iterations=max_nsteps, progress=True):
    if sampler.iteration % autocorr_check_interval == 0:
        tau = sampler.get_autocorr_time(tol=0)
        with open(output_path,"a") as f:
            acceptance_rate = np.mean(sampler.acceptance_fraction)
            print(f"step {sampler.iteration}: autocorr time = {tau}", file=f)
            print(f"acceptance rate = {acceptance_rate:.2f}", file=f)

            flat_samples = sampler.get_chain(flat=True)
            flat_log_probs = sampler.get_log_prob(flat=True)
            best_idx = np.argmax(flat_log_probs)
            current_best = flat_samples[best_idx]
            print(f"current best params = {current_best}", file=f)

        converged = np.all(tau * 100 < sampler.iteration)
        stable = np.all(np.abs(old_tau - tau) / tau < 0.01)

        if converged and stable:
            with open(output_path,"a") as f:
                print(f"converged: step {sampler.iteration}",file=f)
            break

        old_tau = tau

final_tau = sampler.get_autocorr_time()

with open(output_path,"a") as f:
    print(f"final autocorr time: {final_tau}",file=f)
