import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from tqdm import tqdm
import sys
import os
import subprocess
from matplotlib import rcParams

class Sampler():
    def __init__(self,ln_prob,args,ndim):
        self.ln_prob = ln_prob
        self.args = args
        self.ndim = ndim
    
    def calc_ln_prob(self,theta):
        return self.ln_prob(theta,*self.args)
    
    def calc_eigen(self,chain):
        cov = np.cov(chain)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        return eigenvalues, eigenvectors

    def run_metropolis(self,p0,nstep,stepsize,progress=False):
        ln_prob_init = self.calc_ln_prob(p0)

        chain = [[p0[i]] for i in range(self.ndim)]
        ln_prob_list = [ln_prob_init]

        current_params = p0
        current_ln_prob = ln_prob_init
        n_accept = 0
        iterator = tqdm(range(nstep)) if progress else range(nstep)
        for _ in iterator:
            for j in range(self.ndim):
                random_number = np.random.rand()
                step=(random_number-0.5)*stepsize[j]*2
                next_params = current_params.copy()
                next_params[j] += step
                next_ln_prob = self.calc_ln_prob(next_params)
                r = np.random.rand()
                if r < np.exp(next_ln_prob - current_ln_prob):
                    for k in range(self.ndim):
                        chain[k].append(next_params[k])
                    ln_prob_list.append(next_ln_prob)
                    current_params = next_params.copy()
                    current_ln_prob = next_ln_prob.copy()
                    n_accept+=1
                else:
                    for k in range(self.ndim):
                        chain[k].append(current_params[k])
                    ln_prob_list.append(current_ln_prob)
        
        accpetance_ratio = n_accept/nstep/self.ndim
        return np.array(chain) ,np.array(ln_prob_list),accpetance_ratio
    
    def run_metropolis_eigen(self,nstep,chain,prob,progress=False):
        eigen_value, eigen_vector = self.calc_eigen(chain)
        best_ind = np.argmax(prob)
        p0 = list(chain[:,best_ind])
        ln_prob_init = self.calc_ln_prob(p0)

        chain = [[p0[i]] for i in range(self.ndim)]
        ln_prob_list = [ln_prob_init]

        current_params = p0
        current_ln_prob = ln_prob_init
        n_accept = 0
        iterator = tqdm(range(nstep)) if progress else range(nstep)
        for _ in iterator:
            for j in range(self.ndim):
                random_number = np.random.rand()
                step=(random_number-0.5)*np.sqrt(eigen_value[j])*2
                steps = [step*eigen_vector[k,j] for k in range(self.ndim)]
                next_params = [current_params[k]+steps[k] for k in range(self.ndim)]
                next_ln_prob = self.calc_ln_prob(next_params)
                r = np.random.rand()
                if r < np.exp(next_ln_prob - current_ln_prob):
                    for k in range(self.ndim):
                        chain[k].append(next_params[k])
                    ln_prob_list.append(next_ln_prob)
                    current_params = next_params.copy()
                    current_ln_prob = next_ln_prob.copy()
                    n_accept+=1
                else:
                    for k in range(self.ndim):
                        chain[k].append(current_params[k])
                    ln_prob_list.append(current_ln_prob)
        
        accpetance_ratio = n_accept/nstep/self.ndim
        return np.array(chain) ,np.array(ln_prob_list),accpetance_ratio

    def run_metropolis_write(self,p0,nstep,stepsize,out,irun,progress=False):
        ln_prob_init = self.calc_ln_prob(p0)

        current_params = p0
        current_ln_prob = ln_prob_init
        n_accept = 0
        iterator = tqdm(range(nstep)) if progress else range(nstep)
        renzoku = 1

        with open(out,"a") as f:
            for _ in iterator:
                for j in range(self.ndim):
                    random_number = np.random.rand()
                    step=(random_number-0.5)*stepsize[j]*2
                    next_params = current_params.copy()
                    next_params[j] += step
                    next_ln_prob = self.calc_ln_prob(next_params)
                    r = np.random.rand()
                    if r < np.exp(next_ln_prob - current_ln_prob):
                        f.write(f"{irun}  ")
                        for k in range(self.ndim):
                            f.write(f"{current_params[k]}  ")
                        f.write(f"{current_ln_prob} ")
                        f.write(f"{renzoku}  ")
                        f.write("\n")

                        current_params = next_params.copy()
                        current_ln_prob = next_ln_prob.copy()
                        renzoku = 1
                        n_accept+=1
                    else:
                        renzoku += 1
        
        accpetance_ratio = n_accept/nstep/self.ndim

        return accpetance_ratio

    def run_metropolis_eigen_write(self,nstep,out,irun,progress=False):
        chain = [[] for i in range(self.ndim)]
        ln_prob_list = []

        with open(out,"r") as f:
            for line in f:
                line = line.strip().split()
                irun_tmp = int(line[0])
                if irun_tmp != irun-1:
                    continue
                renzoku_tmp = int(line[-1])
                ln_prob_tmp = float(line[-2])
                param_tmp = [float(tmp) for tmp in line[1:-2]]

                for i in range(renzoku_tmp):
                    for k in range(self.ndim):
                        chain[k].append(param_tmp[k])
                    ln_prob_list.append(ln_prob_tmp)

        chain, prob = np.array(chain) ,np.array(ln_prob_list)

        eigen_value, eigen_vector = self.calc_eigen(chain)
        best_ind = np.argmax(prob)
        p0 = list(chain[:,best_ind])
        ln_prob_init = self.calc_ln_prob(p0)

        current_params = p0
        current_ln_prob = ln_prob_init
        n_accept = 0
        iterator = tqdm(range(nstep)) if progress else range(nstep)
        renzoku=1

        with open(out,"a") as f:
            for _ in iterator:
                for j in range(self.ndim):
                    random_number = np.random.rand()
                    step=(random_number-0.5)*np.sqrt(eigen_value[j])*2
                    steps = [step*eigen_vector[k,j] for k in range(self.ndim)]
                    next_params = [current_params[k]+steps[k] for k in range(self.ndim)]
                    next_ln_prob = self.calc_ln_prob(next_params)
                    r = np.random.rand()
                    if r < np.exp(next_ln_prob - current_ln_prob):
                        f.write(f"{irun}  ")
                        for k in range(self.ndim):
                            f.write(f"{current_params[k]}  ")
                        f.write(f"{current_ln_prob} ")
                        f.write(f"{renzoku}  ")
                        f.write("\n")

                        current_params = next_params.copy()
                        current_ln_prob = next_ln_prob.copy()
                        renzoku = 1
                        n_accept+=1
                    else:
                        renzoku += 1
        
        accpetance_ratio = n_accept/nstep/self.ndim
        return accpetance_ratio
    
    def run(self,irun,nsteps,p0,init_step_size,mode="return",out=False,progress=False):
        if mode == "return":
            for i in range(irun):
                if i == 0:
                    class_chain,class_ln_prob_chain,class_accpet = self.run_metropolis(p0,nsteps[i],init_step_size,progress=progress)
                else:
                    class_chain,class_ln_prob_chain,class_accpet = self.run_metropolis_eigen(nsteps[i],class_chain,class_ln_prob_chain,progress=progress)
                print(f"irun= {i}  acceptance ratio = {class_accpet}")
            return class_chain, class_ln_prob_chain

        elif mode == "write":
            if os.path.exists(out):
                confirmation = input(f"{out} is already exist. Do you over write? (yes/no): ")
                if confirmation.lower() == 'yes':
                    try:
                        subprocess.run(['rm', '-rf', out], check=True)
                        print(f"{out} has been removed.")
                    except subprocess.CalledProcessError as e:
                        print(e)
                        sys.exit()
                else:
                    print(f"{out} has not been removed.")
                    sys.exit()

            for i in range(irun):
                if i == 0:
                    class_accpet = self.run_metropolis_write(p0,nsteps[i],init_step_size,out,irun=i,progress=progress)
                else:
                    class_accpet = self.run_metropolis_eigen_write(nsteps[i],out,irun=i,progress=progress)
                print(f"irun= {i}  acceptance ratio = {class_accpet}")
