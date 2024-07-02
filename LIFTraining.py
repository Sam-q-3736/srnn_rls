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
            'gain': 1, # multiplier
            'bias': 10, # mV, bias current
            'v_thr': -55, # mV, threshold
            'v_rest': -65 # mV, resting voltage
        }
    time_params = {
        
    }