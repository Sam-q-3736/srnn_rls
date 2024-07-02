import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from SpikeTraining import SpikeTraining

def create_default_params_LIF():
    neuron_params = {
            'net_size': 300, # units in network
            'tau_s': 100, # ms, slow decay constant
            'tau_f': 2, # ms, fast decay constant
            'tau_m': 10, # ms, membrane decay constant
            'gain': 1.2, # multiplier
            'bias': 10, # mV, bias current
            'v_thr': -55, # mV, threshold
            'v_rest': -65, # mV, resting voltage
            't_refract': 2 # ms, refractory period 
        }
    time_params = {

    }
    connectivity_params = {
            'm': -57/neuron_params['net_size'], # mean
            'std': (17 ** 2)/np.sqrt(neuron_params['net_size']), # standard deviation, 1/sqrt(netsize)
            'cp': 1, # connection probability
        }

class LIFTraining(SpikeTraining): 
    def __init__(self, neuron_params, time_params, train_params, connectivity_params, run_params):
        self.N = neuron_params['net_size']
        self.tau_s = neuron_params['tau_s']
        self.tau_f = neuron_params['tau_f']

        self.slow = np.zeros(self.N)
        self.fast = np.zeros(self.N)
        self.refract = np.zeros(self.N) # time since last refractory period

        # fast and slow connectivity 
        self.Jf = self.genw_sparse(neuron_params['net_size'], connectivity_params['m'], connectivity_params['std'], connectivity_params['cp'])
        self.Js = np.zeros((self.N, self.N))

    def dslow(self): 
        return -1/self.tau_s * self.slow

    def dfast(self):
        return -1/self.tau_f * self.fast
    