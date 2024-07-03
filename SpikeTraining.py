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

    def plot_spk_rasts(spk_rast, inds):
        spk_inds, spk_t = np.nonzero(spk_rast)
        spk_times = []
        for idx in np.unique(spk_inds):
            spk_times.append(spk_t[spk_inds == idx])
        plt.eventplot(spk_times[inds])