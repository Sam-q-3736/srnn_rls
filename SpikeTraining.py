import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

class SpikeTraining:

    def create_default_params():
        raise NotImplementedError

    def __init__(self):
        raise NotImplementedError

    def genw_sparse(self, N, m, std, p):
        weight = sp.stats.norm.rvs(m, std, (N, N))
        is_zero = np.random.choice([0, 1], (N, N), [1-p, p])    
        return np.multiply(weight, is_zero)

    def train_network(self, neuron_params, time_params, train_params, W, stim, targets):
        raise NotImplementedError

    def run_network(self, neuron_params, time_params, W, stim, run_time):
        raise NotImplementedError
    
    def gen_rand_stim(self, pars):
        N = pars['net_size']
        dt = pars['dt']
        timesteps = int(pars['total_time']/dt)
        
        stim = np.zeros((N, timesteps))
        for row in range(N):
            rstim = 2 * sp.stats.uniform.rvs(0, 1) - 1 # random stim weight from -1, 1
            stim[row, int(pars['stim_on']/dt):int(pars['stim_off']/dt)] = rstim
        return stim

    def plot_spk_rasts(spk_rast, inds):
        spk_inds, spk_t = np.nonzero(spk_rast)
        spk_times = []
        for idx in np.unique(spk_inds):
            spk_times.append(spk_t[spk_inds == idx])
        plt.eventplot(spk_times[inds])

    def plot_connectivity_matrix(self): 
        plt.imshow(self.W_trained, cmap=plt.get_cmap('seismic'), vmin = -(max(-1*np.min(self.W_trained), np.max(self.W_trained))), vmax = (max(-1*np.min(self.W_trained), np.max(self.W_trained))))
        plt.title("Connectivity matrix after training")
        plt.colorbar()