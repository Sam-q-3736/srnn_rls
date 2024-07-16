import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

class SpikeTraining:

    def create_default_params():
        raise NotImplementedError

    def __init__(self, pars):
        pass

    def genw_sparse(self, N, m, std, p):
        weight = sp.stats.norm.rvs(m, std, (N, N))
        is_zero = np.random.choice([0, 1], (N, N), [1-p, p])    
        return np.multiply(weight, is_zero)

    def train(self, stim, targets, fout):
        raise NotImplementedError

    def run(self, stim):
        raise NotImplementedError
    
    def gen_rand_stim(self, on, off):
        N = self.N
        dt = self.dt
        timesteps = int(self.T/dt)
        
        stim = np.zeros((N, timesteps))
        for row in range(N):
            rstim = 2 * sp.stats.uniform.rvs(0, 1) - 1 # random stim weight from -1, 1
            stim[row, int(on/dt):int(off/dt)] = rstim
        return stim

    def plot_spk_rasts(self, spk_rast, inds):
        spk_inds, spk_t = np.nonzero(spk_rast)
        spk_times = []
        for idx in np.unique(spk_inds):
            spk_times.append(spk_t[spk_inds == idx])
        plt.eventplot(spk_times[inds])

    def plot_connectivity_matrix(self, mat): 
        plt.imshow(mat, cmap=plt.get_cmap('seismic'), vmin = -(max(-1*np.min(mat), np.max(mat))), vmax = (max(-1*np.min(mat), np.max(mat))))
        plt.title("Connectivity matrix after training")
        plt.colorbar()