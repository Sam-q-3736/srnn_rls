import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from spike_training import *

def create_default_params():
    neuron_params = {
            'net_size': 200, # units in network
            'tau_x': 10 # ms, decay constant
        }
    time_params = {
            'total_time': 1000, # ms, total runtime
            'dt': 0.1, # ms
            'stim_on': 0, # ms
            'stim_off': 50 # ms
        }    
    train_params = {
            'training_loops': 10, # number of training loops
            'train_every': 2 # ms, timestep of updating connectivity matrix
        }
    connectivity_params = {
            'm': 0, # mean
            'std': 1/np.sqrt(200), # standard deviation, 1/sqrt(netsize)
            'cp': 0.3, # connection probability
        }
    run_params = {
            'runtime': 2000 # ms, runtime of trained network
        }
    return neuron_params, time_params, train_params, connectivity_params, run_params

class rate_training(spike_training): 
    def __init__(self, neuron_params, time_params, train_params, connectivity_params, run_params):
        # initialize connectivity matrix
        self.W_init = self.genw_sparse(neuron_params['net_size'], connectivity_params['m'], connectivity_params['std'], connectivity_params['cp'])
        self.W_trained = np.copy(self.W_init)

        # unpack parameters
        self.N = neuron_params['net_size']
        self.tau_x = neuron_params['tau_x']

        self.T = time_params['total_time']
        self.dt = time_params['dt']
        self.stim_on = time_params['stim_on']
        self.stim_off = time_params['stim_off']

        self.nloop = train_params['training_loops']
        self.train_every = train_params['train_every']

        self.run_time = run_params['runtime']

        # track current membrane potential
        self.x = np.zeros(self.N)

    # differential equation of dx/dt
    def dx(self, x, t):
        return 1/tau_x * (-x + self.gain * np.dot(self.W_trained, np.tanh(x)) + fin[:, int(t/self.dt)] + fout[:, int(t/self.dt)])
    
    def rk4_step(self, x, t, fin, fout):
        x1 = self.dt * dx(x)
        x2 = self.dt * dx(x + x1/2)
        x3 = self.dt * dx(x + x2/2)
        x4 = self.dt * dx(x + x3)
        x_next = x + (x1 + 2*x2 + 2*x3 + x4) / 6

        x = x_next

    def run_rate(rate_params, ufin, ufout): 

        # initialize variables to 0 
        # can be excluded to run from previous state
        self.x = np.zeros(self.N)

        # track variables
        x_vals = []

        t = 0
        while t < self.T: 
            # RK4 for each timestep
            self.rk4_step(self.x, t, ufin, ufout)

            t = t + dt
            x_vals.append(self.x)
        
        return np.transpose(x_vals)