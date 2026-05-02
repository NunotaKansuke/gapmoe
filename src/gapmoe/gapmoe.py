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
import warnings


class gapmoe():
    def __init__(self,ra_deg,dec_deg):
        self.KAPPA = 8.1439 
        self.G = 2.959122082855911e-4 
        self.mass_path = "/moao38_7/nunota/gapmoe/ML_hist/ML_hist.dat"
        
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
        gal = coord.galactic
        l = gal.l.deg
        b = gal.b.deg
        
        if l > 180:
            l -= 360
        
        if not (-10 <= l <= 10):
            raise ValueError(f"l={l:.2f} is outside the valid range [-10, 10]")
        if not (-7 <= b <= 6):
            raise ValueError(f"b={b:.2f} is outside the valid range [-7, 6]")

        l_values = np.arange(-10, 10.2, 0.2)
        b_values = np.arange(-7, 6.2, 0.2)

        l_nearest = l_values[np.argmin(np.abs(l_values - l))]
        b_nearest = b_values[np.argmin(np.abs(b_values - b))]

        self.murel_hist_dir = "/moao38_7/nunota/gapmoe/murel_hist"
        self.mu_path = os.path.join(
            self.murel_hist_dir, f"murel_hist_{l_nearest:.1f}_{b_nearest:.1f}.dat"
        )
        
        self.rhos_hist_dir = "/moao38_7/nunota/gapmoe/rhos_hist"
        self.rhos_path = os.path.join(
            self.rhos_hist_dir, f"rhos_hist_{l_nearest:.1f}_{b_nearest:.1f}.dat"
        )
        
        self.comp_names = [
        "thin_disk_0",  # 1
        "thin_disk_1",  # 2
        "thin_disk_2",  # 3
        "thin_disk_3",  # 4
        "thin_disk_4",  # 5
        "thin_disk_5",  # 6
        "thin_disk_6",  # 7
        "thick_disk",   # 8
        "bulge",        # 9
        "NSD",     # 10
        "halo",        # 11
        "all"       #12
        ]
        
        self.rho_comp_names =  ["D"] + self.comp_names
        self.mass_comp_names =  ["M"] + self.comp_names[:-1] + ["fREM"]
        
    def set_data_rho(self):
        self.dl_hist = np.genfromtxt(self.rhos_path, names=self.rho_comp_names, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12],comments='#')
        self.ds_hist = np.genfromtxt(self.rhos_path, names=self.rho_comp_names, usecols=[0,13,14,15,16,17,18,19,20,21,22,23,24],comments='#')

        #Normalization
        dl_total = np.sum(self.dl_hist["all"])
        ds_total = np.sum(self.ds_hist["all"])
        for name in self.comp_names:
            if dl_total > 0:
                self.dl_hist[name] /= dl_total
            if ds_total > 0:
                self.ds_hist[name] /= ds_total
        
        self.dl_cdf= np.cumsum(self.dl_hist["all"])
           
    def set_data_mass(self, fREM_range=[0]):
        self.ml_hist = np.genfromtxt(self.mass_path, names=self.mass_comp_names,comments='#')
        
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

    def set_data(self,fREM_range=[0]):
        self.set_data_rho()
        self.set_data_mass(fREM_range)
        self.set_data_mu()
            
    def get_density_DL(self, DL_value, comp="all"):
        if DL_value <= 16:
            warnings.warn("Warning: DL <= 16. You may be mistaking a value in pc for one in kpc. Please check the unit.")
        if not (0 <= DL_value < len(self.dl_hist["D"])):
            return 0.0
        idx = int(DL_value)
        return self.dl_hist[comp][idx]
    
    def get_density_DL_given_DS(self, DL_value, DS_value):
        if DL_value >= DS_value:
            return 0.0 

        idx_DL = int(DL_value)
        idx_DS = int(DS_value)

        if not (0 <= idx_DL < len(self.dl_hist["D"])) or not (0 <= idx_DS < len(self.ds_hist["D"])):
            return 0.0

        norm_factor = self.dl_cdf[idx_DS - 1] if idx_DS > 0 else 0.0
        if norm_factor == 0.0:
            return 0.0

        return self.dl_hist["all"][idx_DL] / norm_factor

    def get_density_DS(self, DS_value, comp="all"):
        if DS_value <= 16:
            warnings.warn("Warning: DS <= 16. You may be mistaking a value in pc for one in kpc. Please check the unit.")
        if not (0 <= DS_value < len(self.ds_hist["D"])):
            return 0.0
        idx = int(DS_value)
        return self.ds_hist[comp][idx]
    
    def get_density_M(self, M_value, comp="bulge"):
        idx = int((M_value - self.M_MIN) / self.M_BIN_WIDTH)
        if idx < 0 or idx >= len(self.sum_ml_hist["M"]):
            return 0.0

        return self.sum_ml_hist[comp][idx]
    
    def get_density_M_given_DL(self, M_value, DL_value):
        idx_M = int((M_value - self.M_MIN) / self.M_BIN_WIDTH)
        idx_DL = int(DL_value)

        if idx_M < 0 or idx_M >= len(self.sum_ml_hist["M"]):
            return 0.0
        if idx_DL < 0 or idx_DL >= len(self.dl_hist["D"]):
            return 0.0

        p_M_given_DL = 0.0
        total_DL_density = sum([self.dl_hist[comp][idx_DL] for comp in self.valid_components])

        if total_DL_density == 0:
            return 0.0

        for comp in self.valid_components:
            p_M_given_comp = self.sum_ml_hist[comp][idx_M]
            p_comp_given_DL = self.dl_hist[comp][idx_DL] / total_DL_density
            p_M_given_DL += p_M_given_comp * p_comp_given_DL

        return p_M_given_DL

    def get_density_mu_given_DL_DS(self, DL_value, DS_value, mu_value, phi_value):
        if not (self.MU_MIN <= mu_value < self.MU_MAX) or not (self.PHI_MIN <= phi_value < self.PHI_MAX):
            return 0.0, 0.0
        if DS_value <= DL_value:
            return 0.0, 0.0

        i_DL = int((DL_value - self.DL_MIN) / self.DL_WIDTH)
        j_DS = int((DS_value - self.DS_MIN) / self.DS_WIDTH)
        k_mu = int((mu_value - self.MU_MIN) / self.MU_WIDTH)
        l_phi = int((phi_value - self.PHI_MIN) / self.PHI_WIDTH)

        max_len = max(self.MU_BIN_NUM, self.PHI_BIN_NUM)
        DL_offset_start = self.DL_offsets[i_DL]
        DS_offset_start = DL_offset_start + (j_DS - i_DL - 1) * max_len

        mu_line_index = DS_offset_start + k_mu
        phi_line_index = DS_offset_start + l_phi

        mu_hist_value = self.mu_hist[mu_line_index]["mu_hist"]
        phi_hist_value = self.mu_hist[phi_line_index]["phi_hist"]
        return mu_hist_value , phi_hist_value


    def safe_log(self,x):
        if x <= 0:
            return -np.inf
        else:
            return np.log(x)

    def get_joint_log_density(self, ML_value, DL_value, DS_value, mu_value ,phi_value):

        log_p_ML_given_DL = self.safe_log(self.get_density_M_given_DL(ML_value, DL_value))
        log_p_DL_given_DS = self.safe_log(self.get_density_DL_given_DS(DL_value, DS_value))
        log_p_DS = self.safe_log(self.get_density_DS(DS_value))
        
        p_mu_given_DL_DS, p_phi_given_DL_DS= self.get_density_mu_given_DL_DS(DL_value, DS_value, mu_value, phi_value)
        log_p_mu_given_DL_DS, log_p_phi_given_DL_DS = self.safe_log(p_mu_given_DL_DS), self.safe_log(p_phi_given_DL_DS)
        joint_density = log_p_ML_given_DL + log_p_DL_given_DS + log_p_DS + log_p_mu_given_DL_DS + log_p_phi_given_DL_DS

        return joint_density;
    
    def get_joint_log_density_given_DS(self, ML_value, DL_value, DS_value, mu_value ,phi_value):

        log_p_ML_given_DL = self.safe_log(self.get_density_M_given_DL(ML_value, DL_value))
        log_p_DL_given_DS = self.safe_log(self.get_density_DL_given_DS(DL_value, DS_value))
        log_p_DS = self.safe_log(self.get_density_DS(DS_value))
        
        p_mu_given_DL_DS, p_phi_given_DL_DS= self.get_density_mu_given_DL_DS(DL_value, DS_value, mu_value, phi_value)
        log_p_mu_given_DL_DS, log_p_phi_given_DL_DS = self.safe_log(p_mu_given_DL_DS), self.safe_log(p_phi_given_DL_DS)

        joint_density = log_p_ML_given_DL + log_p_DL_given_DS +  log_p_mu_given_DL_DS + log_p_phi_given_DL_DS

        return joint_density
    
    def calc_log_Gamma(self,ML, DL, DS, mu):
        if DL <= 0 or DS <= 0 or DS <= DL:
            return 0.0  # Avoid invalid geometry

        pirel = 1000*((1.0 / DL) - (1.0 / DS))
        thetaE = np.sqrt(ML * pirel * self.KAPPA)
        log_Gamma = 2*self.safe_log(DL) + self.safe_log(thetaE) + self.safe_log(mu)
        return log_Gamma
    
    def log_galactic_prior(self, ML_value, DL_value, DS_value, mu_value, phi_value):
        log_p_joint = self.get_joint_log_density(ML_value, DL_value, DS_value, mu_value, phi_value)
        log_Gamma = self.calc_log_Gamma(ML_value, DL_value, DS_value, mu_value)
        return log_p_joint + log_Gamma
    
    def log_galactic_prior_given_DS(self, ML_value, DL_value, DS_value, mu_value, phi_value):
        log_p_joint = self.get_joint_log_density_given_DS(ML_value, DL_value, DS_value, mu_value, phi_value)
        Gamma = self.calc_Gamma(ML_value, DL_value, DS_value, mu_value)

        if Gamma <= 0.0:
            return -np.inf
        log_Gamma = np.log(Gamma)

        return log_p_joint + log_Gamma
    

    def get_comp_fraction(self,DL_value):
        idx_DL = int(DL_value)

        if idx_DL < 0 or idx_DL >= len(self.dl_hist["D"]):
            return np.zeros(len(self.valid_components))

        total_DL_density = sum([self.dl_hist[comp][idx_DL] for comp in self.valid_components])

        if total_DL_density == 0:
            return np.zeros(len(self.valid_components))

        comp_fractions = []
        for comp in self.valid_components:
            p_comp_given_DL = self.dl_hist[comp][idx_DL] / total_DL_density
            comp_fractions.append(p_comp_given_DL)

        return np.array(comp_fractions)
