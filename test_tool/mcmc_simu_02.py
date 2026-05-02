#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
gapmoe_model = gapmoe(rhos_path="/moao38_7/nunota/gapmoe/test_data/hist_D.dat",mass_path="/moao38_7/nunota/gapmoe/test_data/hist_M.dat",mu_path="/moao38_7/nunota/gapmoe/test_data/hist_murel.dat")
gapmoe_model.set_data()

JD0 = 2450000

args = sys.argv
for i in range(len(args)):
  if args[i]  == "-output":
    output_path = args[i+1]
  if args[i]  == "-save":
    backend_path = args[i+1]
  if args[i]  == "-seed":
    seed = int(args[i+1])

# In[5]:


tref = 10063.874
coords = "17:57:38.03 -28:38:28.53"

VBM.t0_par = tref+JD0
VBM.parallaxsystem = 1
VBM.SetObjectCoordinates(coords)


# In[6]:


def mag2flux(mag):
    return 10**(-mag / 2.5)

def flux2mag(flux):
    return -2.5 * np.log10(flux)

def linear_fit(x,y,w):
    w_sum = np.sum(w)
    wxy_sum = np.sum(w*x*y)
    wx_sum = np.sum(w*x)
    wy_sum = np.sum(w*y)
    wxx_sum = np.sum(w*x*x)
    bunbo = w_sum*wxx_sum-wx_sum**2
    a = (w_sum*wxy_sum-wx_sum*wy_sum)/bunbo
    b = (wxx_sum*wy_sum-wx_sum*wxy_sum)/bunbo
    y_fit = a * x + b
    chi2 = np.sum(w * (y - y_fit) ** 2)
    return a,b,chi2

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


# In[7]:


loaded = np.load("/moao38_7/nunota/gapmoe/simu_data/simu_data_02.npz", allow_pickle=True)

simu_data = {k: loaded[k] for k in loaded.files}

Is = 20 + AI
Vs = 1 + 20 + EVI
fs_I = mag2flux(Is)
fs_V = mag2flux(Vs)

tE = 68.8752
rho = 8.9597e-05
s = 0.950293
piEN = 0.132695
piEE = 0.132695
gamma1 = 0.000702124
gamma2 = 0.00175532
gamma3 = 0.00175532

t0 = 10090 - 5       
u0 = 0.01          
q = 0.005
alpha = 3.75*np.pi/4


init_params = np.array([t0,tE,u0,q,s,alpha,rho,piEN,piEE,gamma1,gamma2,gamma3])

# In[12]:


kappa = 8.144; #mas /MO
G = 2.959122082855911e-4 
@jit
def calc_physical_params(x,thS):
    tE, rho, s, piEN, piEE, gamma1, gamma2, gamma3 = x
    piE = jnp.sqrt(piEE**2 + piEN**2)
    
    thE = thS / rho
    ML = thE / kappa / piE
    murel = thE / tE * 365.25
    murel_E = murel * piEE / piE
    murel_N = murel * piEN / piE

    gamma_sq = gamma1**2 + gamma2**2 + gamma3**2
    gamma_parallel = jnp.sqrt(gamma1**2 + gamma3**2)
    gamma_ratio = gamma_parallel / gamma3
    orbital_scale = jnp.cbrt((s**3) * gamma_sq * gamma_ratio / (ML * G))
    gamma_abs = jnp.sqrt(gamma_sq)
    Ds = 1 / ((orbital_scale - piE) * thE)

    pi_rel = thE * piE
    pi_S = 1 / Ds
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L

    RE = DL * thE
    orbital_radi = RE * s * gamma_parallel / gamma3

    cosi = gamma3 * gamma2 / (gamma_parallel * gamma_abs)
    tanphi = - gamma1 * gamma_abs / (gamma3 * gamma_parallel)

    return jnp.array([murel_E, murel_N, ML, DL, Ds, orbital_radi, cosi, tanphi])

@jit
def calc_ln_det_jacobian(x, thS):
    J = jacfwd(calc_physical_params)(x, thS)
    _, lndet = jnp.linalg.slogdet(J)
    return lndet


# In[19]:


physical_use_ind = np.array([1,6,4,7,8,9,10,11])

def calc_ln_like(theta,source_flux=False):
    _t0,_tE,_u0,_q,_s,_theta,_rho = theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6]
    _piEN, _piEE,_gamma1, _gamma2, _gamma3 = theta[7],theta[8],theta[9],theta[10],theta[11]
    _pr = [math.log(_s), math.log(_q), _u0, _theta, math.log(_rho), math.log(_tE), _t0+JD0, _piEN, _piEE, _gamma1,_gamma2,_gamma3]

    _chi2_sum = 0
    for _name in simu_data.keys():
        _model_amp = np.array(VBM.BinaryLightCurveOrbital(_pr,simu_data[_name]["time"]+JD0)[0])
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
    
def calc_ln_prior_gal(x, thS):
    murel_E, murel_N, ML, DL, DS, orbital_radi, cosi, tanphi = calc_physical_params(x, thS)
    mu_abs = jnp.sqrt(murel_E**2 + murel_N**2)
    mu_phi = jnp.arctan2(murel_E,murel_N)
    DL *= 1e3
    DS *= 1e3
    
    if not (gapmoe_model.M_MIN <= ML <= gapmoe_model.M_MAX):
        return -np.inf,-np.inf
    if not (gapmoe_model.DL_MIN <= DL < gapmoe_model.DL_MAX):
        return -np.inf,-np.inf
    if not (gapmoe_model.DS_MIN <= DS < gapmoe_model.DS_MAX):
        return -np.inf,-np.inf
    if DL >= DS:
        return -np.inf,-np.inf
    if not (gapmoe_model.MU_MIN <= mu_abs < gapmoe_model.MU_MAX):
        return -np.inf,-np.inf
    if (orbital_radi < 0.0) or (orbital_radi > 10000.0):
        return -np.inf,-np.inf
    
    return gapmoe_model.log_galactic_prior(ML, DL, DS, mu_abs, mu_phi),1/(1+tanphi**2) 

def check_positive(theta):
    _t0, _tE, _u0, _q, _s, _theta, _rho = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6]
    return _s > 0 and _q > 0 and _rho > 0 and _tE > 0

def ln_prob(theta):
    if not check_positive(theta):
        return -np.inf, np.nan
    _use_params = theta[physical_use_ind]
    _ln_like, _fs_i, _fs_v = calc_ln_like(theta,source_flux=True)
    _thS = calc_thS(_fs_i,_fs_v)
    _ln_det = calc_ln_det_jacobian(_use_params,_thS)
    _ln_prior_gal, _weight_tan = calc_ln_prior_gal(_use_params,_thS)
        
    if jnp.isneginf(_ln_like) or np.isneginf(_ln_det) or jnp.isneginf(_ln_prior_gal):
        return -jnp.inf, np.nan

    ln_post = _ln_like + _ln_prior_gal + _ln_det + np.log(_weight_tan)
    return ln_post, _thS

# In[32]:


param_stds = [
    0.01,     # t0 [days]
    0.1,      # tE
    0.005,    # u0
    0.0001,    # q
    0.01,     # s
    0.05,     # alpha [rad]
    3e-4,     # rho
    0.00001,     # piEN
    0.00001,     # piEE
    3e-4,     # gamma1
    3e-4,     # gamma2
    3e-4      # gamma3
]

ndim = len(init_params)
nwalkers = 48
max_nsteps = 10000
autocorr_check_interval = 100 

np.random.seed(45)
p0 = np.array([
    np.array(init_params) + np.random.normal(scale=param_stds)
    for _ in range(nwalkers)
])

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
