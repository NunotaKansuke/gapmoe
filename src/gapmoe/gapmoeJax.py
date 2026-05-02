import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from jax import jit
from jax import vmap
from functools import partial
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

class gapmoe_jax:
    def __init__(self, ra_deg, dec_deg):
        self.KAPPA = 8.1439 
        self.G = 2.959122082855911e-4 

        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
        gal = coord.galactic
        l = gal.l.deg
        b = gal.b.deg
        
        if l > 180:
            l -= 360
        
        if not (-10 <= l <= 10):
            raise ValueError(f"l={l:.2f} is outside [-10,10]")
        if not (-7 <= b <= 6):
            raise ValueError(f"b={b:.2f} is outside [-7,6]")
        
        l_values = np.arange(-10, 10.2, 0.2)
        b_values = np.arange(-7, 6.2, 0.2)
        l_nearest = l_values[np.argmin(np.abs(l_values - l))]
        b_nearest = b_values[np.argmin(np.abs(b_values - b))]

        base = "/moao38_7/nunota/gapmoe"
        self.mass_path = f"{base}/ML_hist/ML_hist.dat"
        self.murel_hist_dir = f"{base}/murel_hist"
        self.rhos_hist_dir = f"{base}/rhos_hist"
        self.mu_path = f"{self.murel_hist_dir}/murel_hist_{l_nearest:.1f}_{b_nearest:.1f}.dat"
        self.rhos_path = f"{self.rhos_hist_dir}/rhos_hist_{l_nearest:.1f}_{b_nearest:.1f}.dat"

        self.comp_names = [
            "thin_disk_0","thin_disk_1","thin_disk_2","thin_disk_3",
            "thin_disk_4","thin_disk_5","thin_disk_6",
            "thick_disk","bulge","NSD","halo","all"
        ]
        self.rho_comp_names =  ["D"] + self.comp_names
        self.mass_comp_names = ["M"] + self.comp_names[:-1] + ["fREM"]

    # ========== データ読み込み ==========
    def set_data_rho(self):
        self.dl_hist = np.genfromtxt(self.rhos_path, names=self.rho_comp_names, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12], comments='#')
        self.ds_hist = np.genfromtxt(self.rhos_path, names=self.rho_comp_names, usecols=[0,13,14,15,16,17,18,19,20,21,22,23,24], comments='#')
        dl_total = np.sum(self.dl_hist["all"])
        ds_total = np.sum(self.ds_hist["all"])
        for name in self.comp_names:
            if dl_total > 0:
                self.dl_hist[name] /= dl_total
            if ds_total > 0:
                self.ds_hist[name] /= ds_total
        self.dl_cdf = np.cumsum(self.dl_hist["all"])
        self.dl_hist = {k: jnp.array(self.dl_hist[k]) for k in self.dl_hist.dtype.names}
        self.ds_hist = {k: jnp.array(self.ds_hist[k]) for k in self.ds_hist.dtype.names}
        self.dl_cdf = jnp.array(self.dl_cdf)

    def set_data_mass(self, fREM_range=[0]):
        self.ml_hist = np.genfromtxt(self.mass_path, names=self.mass_comp_names, comments='#')
        with open(self.mass_path, 'r') as f:
            first_line = f.readline().strip().split()
            self.M_MIN, self.M_MAX, self.M_BIN_WIDTH = float(first_line[2]), float(first_line[4]), float(first_line[6])
        self.sum_ml_hist = {"M": self.ml_hist["M"][np.where(self.ml_hist['fREM'] == fREM_range[0])[0]]}
        self.valid_components = []
        for comp in self.comp_names[:-1]:
            sum_counts = np.zeros_like(self.ml_hist[comp][np.where(self.ml_hist['fREM'] == fREM_range[0])[0]])
            for f in fREM_range:
                ind_f = np.where(self.ml_hist['fREM'] == f)[0]
                sum_counts += self.ml_hist[comp][ind_f]
            total_counts = np.sum(sum_counts)
            if total_counts == 0:
                pdf = np.zeros_like(sum_counts)
            else:
                pdf = sum_counts / (total_counts * self.M_BIN_WIDTH)
                self.valid_components.append(comp)
            self.sum_ml_hist[comp] = pdf
        self.ml_hist = {k: jnp.array(self.ml_hist[k]) for k in self.ml_hist.dtype.names}
        self.sum_ml_hist = {k: jnp.array(v) for k, v in self.sum_ml_hist.items()}

    def set_data_mu(self):
        self.mu_hist = np.genfromtxt(self.mu_path,names=['DS', 'DL', 'mu', 'mu_hist', 'phi', 'phi_hist'],comments='#')

        with open(self.mu_path, "r") as f:
            first_line = f.readline().strip()

            if first_line.startswith("# WARNING:"):
                first_line = f.readline().strip()

            first_line = first_line.split()
                
            self.MU_MIN = float(first_line[3])
            self.MU_MAX = float(first_line[5])
            self.MU_WIDTH = float(first_line[7])
            self.MU_BIN_NUM = int(first_line[9])

            self.PHI_MIN = float(first_line[11])
            self.PHI_MAX = float(first_line[13])
            self.PHI_WIDTH = float(first_line[15])
            self.PHI_BIN_NUM = int(first_line[17])

            self.DL_MIN = float(first_line[19])
            self.DL_MAX = float(first_line[21])
            self.DL_WIDTH = float(first_line[23])
            self.DL_BIN_NUM = int(first_line[25])

            self.DS_MIN = float(first_line[27])
            self.DS_MAX = float(first_line[29])
            self.DS_WIDTH = float(first_line[31])
            self.DS_BIN_NUM = int(first_line[33])
            
        self.DL_offsets = []
        cumulative_offset = 0
        max_len = max(self.MU_BIN_NUM, self.PHI_BIN_NUM)
        for i_DL in range(self.DL_BIN_NUM):
            valid_DS_count = self.DS_BIN_NUM - i_DL - 1
            self.DL_offsets.append(cumulative_offset)
            cumulative_offset += valid_DS_count * max_len
        unique_DL_DS = set(zip(self.mu_hist['DL'], self.mu_hist['DS']))
        for dl_val, ds_val in unique_DL_DS:
            if ds_val <= dl_val:
                continue
            mask_mu = (self.mu_hist['DL'] == dl_val) & (self.mu_hist['DS'] == ds_val) & (self.mu_hist['mu'] != 0)
            mask_phi = (self.mu_hist['DL'] == dl_val) & (self.mu_hist['DS'] == ds_val) & (self.mu_hist['phi'] != 0)
            mu_sum = np.sum(self.mu_hist['mu_hist'][mask_mu])
            phi_sum = np.sum(self.mu_hist['phi_hist'][mask_phi])
            if mu_sum > 0.0:
                self.mu_hist['mu_hist'][mask_mu] /= mu_sum * self.MU_WIDTH
            if phi_sum > 0.0:
                self.mu_hist['phi_hist'][mask_phi] /= phi_sum * self.PHI_WIDTH
        self.mu_hist = {k: jnp.array(self.mu_hist[k]) for k in self.mu_hist.dtype.names}
        self.DL_offsets = jnp.array(self.DL_offsets)

    def set_data(self, fREM_range=[0]):
        self.set_data_rho()
        self.set_data_mass(fREM_range)
        self.set_data_mu()


    # ========== 基本関数 ==========
    @staticmethod
    def safe_log(x):
        return jnp.where(x <= 0, -jnp.inf, jnp.log(x))

    def calc_log_Gamma(self,ML, DL, DS, mu, KAPPA):
        valid = (DL > 0) & (DS > 0) & (DS > DL)
        pirel = 1000.0 * ((1.0 / DL) - (1.0 / DS))
        thetaE = jnp.sqrt(ML * pirel * KAPPA)
        log_Gamma = 2*self.safe_log(DL) + self.safe_log(thetaE) + self.safe_log(mu)
        return jnp.where(valid, log_Gamma, 0.0)

    # ========== 密度関数 (簡易JAX化版) ==========
    def get_density_M_given_DL(self, M_value, DL_value):
        idx_M = jnp.clip(jnp.floor((M_value - self.M_MIN) / self.M_BIN_WIDTH).astype(int), 0, len(self.sum_ml_hist["M"]) - 1)
        idx_DL = jnp.clip(jnp.floor(DL_value).astype(int), 0, len(self.dl_hist["D"]) - 1)
        p_M_given_comp = jnp.array([self.sum_ml_hist[comp][idx_M] for comp in self.valid_components])
        dl_densities = jnp.array([self.dl_hist[comp][idx_DL] for comp in self.valid_components])
        total_DL_density = jnp.sum(dl_densities)
        p_comp_given_DL = jnp.where(total_DL_density == 0.0, jnp.zeros_like(dl_densities), dl_densities / total_DL_density)
        p_M_given_DL = jnp.sum(p_M_given_comp * p_comp_given_DL)
        return p_M_given_DL

    def get_density_DS(self, DS_value, comp="all"):
        idx = jnp.clip(jnp.floor(DS_value).astype(int), 0, len(self.ds_hist["D"]) - 1)
        return self.ds_hist[comp][idx]

    def get_density_DL_given_DS(self, DL_value, DS_value):
        idx_DL = jnp.clip(jnp.floor(DL_value).astype(int), 0, len(self.dl_hist["D"]) - 1)
        idx_DS = jnp.clip(jnp.floor(DS_value).astype(int), 0, len(self.ds_hist["D"]) - 1)
        valid = DS_value > DL_value
        norm_factor = jnp.where(idx_DS > 0, self.dl_cdf[idx_DS - 1], 0.0)
        val = jnp.where(valid & (norm_factor > 0.0), self.dl_hist["all"][idx_DL] / norm_factor, 0.0)
        return val
    
    def get_density_mu_given_DL_DS(self, DL_value, DS_value, mu_value, phi_value):
        valid = (DS_value > DL_value) & (self.MU_MIN <= mu_value) & (mu_value < self.MU_MAX) & (self.PHI_MIN <= phi_value) & (phi_value < self.PHI_MAX)
        i_DL = jnp.clip(jnp.floor((DL_value - self.DL_MIN) / self.DL_WIDTH).astype(int), 0, self.DL_BIN_NUM - 1)
        j_DS = jnp.clip(jnp.floor((DS_value - self.DS_MIN) / self.DS_WIDTH).astype(int), 0, self.DS_BIN_NUM - 1)
        k_mu = jnp.clip(jnp.floor((mu_value - self.MU_MIN) / self.MU_WIDTH).astype(int), 0, self.MU_BIN_NUM - 1)
        l_phi = jnp.clip(jnp.floor((phi_value - self.PHI_MIN) / self.PHI_WIDTH).astype(int), 0, self.PHI_BIN_NUM - 1)
        max_len = jnp.maximum(self.MU_BIN_NUM, self.PHI_BIN_NUM)
        DL_offset_start = self.DL_offsets[i_DL]
        DS_offset_start = DL_offset_start + (j_DS - i_DL - 1) * max_len
        mu_line_index = DS_offset_start + k_mu
        phi_line_index = DS_offset_start + l_phi
        mu_hist_value = jnp.where(valid, self.mu_hist["mu_hist"][mu_line_index], 0.0)
        phi_hist_value = jnp.where(valid, self.mu_hist["phi_hist"][phi_line_index], 0.0)
        return mu_hist_value, phi_hist_value


    # ========== joint log density ==========
    def get_joint_log_density(self, ML_value, DL_value, DS_value, mu_value, phi_value):
        log_p_ML_given_DL = self.safe_log(self.get_density_M_given_DL(ML_value, DL_value))
        log_p_DL_given_DS = self.safe_log(self.get_density_DL_given_DS(DL_value, DS_value))
        log_p_DS = self.safe_log(self.get_density_DS(DS_value))
        p_mu, p_phi = self.get_density_mu_given_DL_DS(DL_value, DS_value, mu_value, phi_value)
        log_p_mu = self.safe_log(p_mu)
        log_p_phi = self.safe_log(p_phi)
        return log_p_ML_given_DL + log_p_DL_given_DS + log_p_DS + log_p_mu + log_p_phi


    # ========== main prior ==========
    @partial(jit, static_argnums=(0,))
    def log_galactic_prior(self, ML_value, DL_value, DS_value, mu_value, phi_value):
        log_p_joint = self.get_joint_log_density(ML_value, DL_value, DS_value, mu_value, phi_value)
        log_Gamma = self.calc_log_Gamma(ML_value, DL_value, DS_value, mu_value, self.KAPPA)
        return log_p_joint + log_Gamma
    
    @partial(jit, static_argnums=(0,))
    def grad_log_galactic_prior(self, ML_value, DL_value, DS_value, mu_value, phi_value, argnums=0):
        def wrapped(ML, DL, DS, mu, phi):
            return self.log_galactic_prior(ML, DL, DS, mu, phi)
        grad_fn = jax.grad(wrapped, argnums=argnums)
        return grad_fn(ML_value, DL_value, DS_value, mu_value, phi_value)
    
    def get_comp_fraction(self, DL_value):
        idx_DL = jnp.clip(jnp.floor(DL_value).astype(int), 0, len(self.dl_hist["D"]) - 1)

        densities = jnp.array([self.dl_hist[comp][idx_DL] for comp in self.valid_components])
        total_DL_density = jnp.sum(densities)

        comp_fractions = jnp.where(total_DL_density == 0,
                                   jnp.zeros_like(densities),
                                   densities / total_DL_density)
        return comp_fractions
