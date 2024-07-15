import numpy as np
import scipy as sp
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from RateTraining import RateTraining

def create_default_params_rate():
    p = {
            'net_size': 300, # units in network
            'tau_x': 10, # ms, decay constant
            'gain': 1, # multiplier
            'total_time': 2000, # ms, total runtime
            'dt': 1, # ms
            'stim_on': 0, # ms
            'stim_off': 0, # ms, matches run-forward time in FF_Demo
            'lam': 1, # learning rate factor
            'training_loops': 20, # number of training loops
            'train_every': 2, # ms, timestep of updating connectivity matrix
            'm': 0, # mean
            'std': 1, # standard deviation multiplier, 1/sqrt(netsize)
            'cp': 1, # connection probability
            'runtime': 2000 # ms, runtime of trained network
        }
    return p

class RateTrainingGPU(RateTraining): 
    def __init__(self, p):
        # initialize connectivity matrix
        self.W_init = self.genw_sparse(p['net_size'], p['m'], p['std']/np.sqrt(p['net_size']), p['cp'])
        self.W_trained = np.copy(self.W_init)
        # initalize output weights
        self.W_out = np.zeros(p['net_size'])

        # unpack parameters
        self.N = p['net_size']
        self.tau_x = p['tau_x']
        self.gain = p['gain']

        self.T = p['total_time']
        self.dt = p['dt']
        self.stim_on = p['stim_on']
        self.stim_off = p['stim_off']

        self.lam = p['lam']
        self.nloop = p['training_loops']
        self.train_every = p['train_every']

        self.run_time = p['runtime']

        # track current activity
        self.x = np.zeros(self.N)
        self.Hx = np.tanh(self.x)

    def toGPU(self):
        self.W_trained = cp.asarray(self.W_trained)
        self.W_out = cp.asarray(self.W_out)
        self.x = cp.asarray(self.x)
        self.Hx = cp.asarray(self.Hx)

    def toCPU(self):
        self.W_trained = cp.asnumpy(self.W_trained)
        self.W_out = cp.asnumpy(self.W_out)
        self.x = cp.asnumpy(self.x)
        self.Hx = cp.asnumpy(self.Hx)

    def stepGPU(self, stim, itr):
        ext = stim[:, itr]

        self.x = self.x + self.dt * 1/self.tau_x * (-self.x + self.gain * cp.dot(self.W_trained, cp.tanh(self.x)) + ext)
        self.Hx = cp.tanh(self.x)

    def run_rateGPU(self, stim): 

        #t = 0
        itr = 0
        timesteps = int(self.T/self.dt)
        
        # track variables
        x_vals = cp.zeros((timesteps, self.N))
        Hx_vals = cp.zeros((timesteps, self.N))

        self.toGPU()
        stim = cp.asarray(stim)

        while itr < timesteps:
            # RK4 for each timestep
            #self.rk4_step(stim, itr)
            self.stepGPU(stim, itr)

            # track variables
            x_vals[itr, :] = self.x
            Hx_vals[itr, :] = self.Hx

            #t = t + self.dt
            itr = itr + 1
                
        x_vals = np.transpose(x_vals)
        Hx_vals = np.transpose(Hx_vals)

        self.toCPU()
        x_vals = cp.asnumpy(x_vals)
        Hx_vals = cp.asnumpy(Hx_vals)

        return x_vals, Hx_vals