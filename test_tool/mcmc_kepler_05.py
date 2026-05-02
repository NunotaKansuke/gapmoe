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

def _wrap_pi(x):  # [-pi,pi]
    return (x + jnp.pi) % (2*jnp.pi) - jnp.pi

def _kepler_to_state_NE(a_AU, e, cos_i, Om_NE, om, nu, G, ML):
    i = jnp.arccos(cos_i)

    r_0 = a_AU*(1-e*e)/(1 + e * jnp.cos(nu))
    # perifocal frame
    r_pf = jnp.array([r_0 * jnp.cos(nu),
                      r_0 * jnp.sin(nu),
                      0.0])
    n = jnp.sqrt(G*ML/a_AU**3)
    v_pf = jnp.array([-n*a_AU*jnp.sin(nu)/jnp.sqrt(1-e*e),
                       n*a_AU*(e+jnp.cos(nu))/jnp.sqrt(1-e*e),
                       0.0])

    cO,sO = jnp.cos(Om_NE), jnp.sin(Om_NE)
    co,so = jnp.cos(om), jnp.sin(om)
    ci,si = cos_i, jnp.sqrt(1-cos_i**2)
    Rz_Om = jnp.array([[ cO,-sO,0.0],
                       [ sO, cO,0.0],
                       [0.0,0.0,1.0]])
    Rx_i  = jnp.array([[1.0,0.0,0.0],
                       [0.0, ci,-si],
                       [0.0, si, ci]])
    Rz_om = jnp.array([[ co,-so,0.0],
                       [ so, co,0.0],
                       [0.0,0.0,1.0]])
    R = Rz_Om @ Rx_i @ Rz_om
    r = R @ r_pf
    v = R @ v_pf
    return r, v

thS = 0.003124442
def inverse_lightcurve_params_from_physical(
    phys, thS, vEarth_NE,
    G=2.959122082855911e-4,   # AU^3 / (Msun * day^2)
    KAPPA=8.1429              # mas / Msun
):
    """
    phys = [t0, u0, q, ML, DL, DS, muN_hel, muE_hel, a_AU, e, cos_i, Om_NE, om, nu]
    return: theta = [t0, tE, u0, rho, q, s, alpha,
                     piEN, piEE, gamma1, gamma2, gamma3, r_s, a_s]
    """
    (t0, u0, q, ML, DL, DS, muN_hel, muE_hel,
     a_AU, e, cos_i, Om_NE, om, nu) = phys

    # 1) NEで r,v
    r_AU, v_AU_d = _kepler_to_state_NE(a_AU, e, cos_i, Om_NE, om, nu, G, ML)

    # 2) θE, πE, RE
    pi_S = 1.0/DS         # 1/kpc
    pi_L = 1.0/DL
    pi_rel = pi_L - pi_S  # 1/kpc
    thE = jnp.sqrt(KAPPA * ML * pi_rel)   # mas
    piE = pi_rel / thE                     # dimensionless
    RE  = DL * thE                         # AU  (1 mas × 1 kpc = 1 AU)

    # 3) μ_geo と tE
    muN_geo = muN_hel - thE * piE * vEarth_NE[0]
    muE_geo = muE_hel - thE * piE * vEarth_NE[1]
    mu_geo  = jnp.sqrt(muN_geo**2 + muE_geo**2)  # mas/yr
    tE = thE / (mu_geo / 365.25)                 # day

    # πE の向き(μ_geo に合わせる)
    ang_mu = jnp.arctan2(muE_geo, muN_geo)
    piEN = piE * jnp.cos(ang_mu)
    piEE = piE * jnp.sin(ang_mu)

    # 4) binary axix 座標の基底（rで定義）
    z_hat = jnp.array([0.0, 0.0, 1.0]) # LOS
    p = r_AU - jnp.dot(r_AU, z_hat) * z_hat  #天球面へ射影したr_AUベクトル
    p_norm = jnp.linalg.norm(p)
    eps = 1e-300
    xB = p / (p_norm + eps)
    yB = jnp.cross(z_hat, xB)
    yB = yB / (jnp.linalg.norm(yB) + eps)       # 念のため正規化

    # 5) s, r_s, gammas
    s = p_norm / (RE + eps)
    r_s = (jnp.dot(r_AU, z_hat)) / ((RE + eps) * (s + eps))
    gamma1 = jnp.dot(v_AU_d, xB) / ((RE + eps) * (s + eps))
    gamma2 = jnp.dot(v_AU_d, yB) / ((RE + eps) * (s + eps))
    gamma3 = jnp.dot(v_AU_d, z_hat) / ((RE + eps) * (s + eps))

    # 6) a_s
    a_scaled = a_AU / (RE + eps)
    a_s = a_scaled / (s * jnp.sqrt(1 + r_s**2) + eps)

    # 7) alpha
    h = jnp.cross(
        jnp.array([p_norm, 0, r_AU[2]]),
        jnp.array([
            jnp.dot(v_AU_d, xB),
            jnp.dot(v_AU_d, yB),
            jnp.dot(v_AU_d, z_hat)
        ])
    ) # angular momentum in binary axis  coordinate

    z_orb = h / (jnp.linalg.norm(h) + eps)
    Om0 = jnp.arctan2(z_orb[0], -z_orb[1])
    alpha = Om0 + jnp.arctan2(piEE, piEN) - Om_NE
    alpha = _wrap_pi(alpha)

    rho = thS / thE

    theta = jnp.array([t0, tE, u0, rho, q, s, alpha,
                       piEN, piEE, gamma1, gamma2, gamma3, r_s, a_s])
    return theta

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


loaded = np.load("/moao38_7/nunota/gapmoe/simu_data/simu_data_05_kepler.npz", allow_pickle=True)

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

t0 = 10090
u0 = -0.02
q = 0.01
ML = 0.7
DL = 1.5
DS = 8.1
muN_hel = 4
muE_hel = -4
a_AU = 2.3
e = 0.2
cos_i = 0.1
Om_NE = 0.1 - np.pi
om = -1
nu = -2

phys_base = np.array([
    t0,       # 0
    u0,       # 1
    q,        # 2
    ML,       # 3
    DL,       # 4
    DS,       # 5
    muN_hel,  # 6
    muE_hel,  # 7
    a_AU,     # 8
    e,        # 9
    cos_i,    # 10
    Om_NE,    # 11
    om,       # 12
    nu,       # 13
])

phys_stds = np.array([
    0.01,     # t0     ±0.5 day
    0.001,    # u0
    0.0002,   # q
    0.005,    # ML
    0.01,     # DL
    0.01,     # DS
    0.05,     # muN_hel
    0.05,     # muE_hel
    0.2,     # a_AU
    0.02,    # e
    0.01,    # cos_i
    0.01,    # Om_NE
    0.01,    # om
    0.01,    # nu
])


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

def read_chain(path,burnin,thin,flat=True):
    sampler = emcee.backends.HDFBackend(path)
    chain = sampler.get_chain(flat=flat, discard=burnin, thin=thin)
    blob = sampler.get_blobs(flat=flat, discard=burnin, thin=thin)
    lnprob = sampler.get_log_prob(flat=flat, discard=burnin, thin=thin)
    return chain, blob,lnprob
# In[7]:

param_stds = [
    0.001,     # t0 [days]
    0.0001,      # tE
    0.00001,    # u0
    3e-8,     # rho
    0.0000001,    # q
    0.000001,     # s
    0.00005,     # alpha [rad]
    0.0001,     # piEN
    0.0001,     # piEE
    5e-10,     # gamma1
    5e-10,     # gamma2
    5e-10,      # gamma3
    1e-4,
    1e-4
]

ndim = len(init_params)
nwalkers = 28
max_nsteps = 1000000
autocorr_check_interval = 100 

np.random.seed(45)
p0 = np.array([
    np.array(init_params) + np.random.normal(scale=param_stds)
    for _ in range(nwalkers)
])
p0 = np.load("/moao38_7/nunota/gapmoe/init_params/event05_gap_3.npy")


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
