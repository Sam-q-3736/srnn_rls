import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from spike_training import *

def create_default_params():
    neuron_params = {
            'net_size': 200, # units in network
            'tau_x': 10, # ms, decay constant
            'gain': 1.2, # multiplier
        }
    time_params = {
            'total_time': 1000, # ms, total runtime
            'dt': 0.1, # ms
            'stim_on': 0, # ms
            'stim_off': 50 # ms
        }    
    train_params = {
            'lam': 1, # learning rate factor
            'training_loops': 10, # number of training loops
            'train_every': 2 # ms, timestep of updating connectivity matrix
        }
    connectivity_params = {
            'm': 0, # mean
            'std': 1/np.sqrt(200), # standard deviation, 1/sqrt(netsize)
            'cp': 1, # connection probability
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
        self.gain = neuron_params['gain']

        self.T = time_params['total_time']
        self.dt = time_params['dt']
        self.stim_on = time_params['stim_on']
        self.stim_off = time_params['stim_off']

        self.lam = train_params['lam']
        self.nloop = train_params['training_loops']
        self.train_every = train_params['train_every']

        self.run_time = run_params['runtime']

        # track current activity
        self.x = np.zeros(self.N)
        self.Hx = np.tanh(self.x)

    # differential equation of dx/dt
    def dx(self, x, ext, itr):
        return 1/self.tau_x * (-x + self.gain * np.dot(self.W_trained, self.Hx) + ext)
    
    def rk4_step(self, stim, itr):
        ext = stim[:, itr]

        x1 = self.dt * self.dx(self.x, ext, itr)
        x2 = self.dt * self.dx(self.x + x1/2, ext, itr)
        x3 = self.dt * self.dx(self.x + x2/2, ext, itr)
        x4 = self.dt * self.dx(self.x + x3, ext, itr)
        x_next = self.x + (x1 + 2*x2 + 2*x3 + x4) / 6

        self.x = x_next
        self.Hx = np.tanh(self.x)

    def train_rate(self, stim, targets):
        # initialize network activity to 0 
        # can be excluded to run from previous state
        self.x = np.zeros(self.N)
        self.Hx = np.tanh(self.x)

        # initialize correlation matrix
        P = np.eye(self.N) * 1/self.lam

        # track variables
        x_vals = []
        Hx_vals = []

        t = 0
        itr = 0
        timesteps = int(self.T/self.dt)

        # training loop
        for i in range(self.nloop):
            print('training trial', i)
            t = 0
            itr = 0
            
            while itr < timesteps: 
                
                # calculate next step of diffeqs
                self.rk4_step(stim, itr)

                # update timestep
                t = t + self.dt
                itr = itr + 1

                # track variables
                x_vals.append(self.x)
                Hx_vals.append(self.Hx)

                # train connectivity matrix
                if itr > int(self.stim_off/self.dt) and itr < timesteps \
                    and np.mod(itr, int(self.train_every/self.dt)) == 0:
                    
                    # update correlation matrix
                    numer = np.outer(np.dot(P, self.Hx), np.dot(P, self.Hx))
                    denom = 1 + np.dot(self.Hx, np.dot(P, self.Hx))
                    P = P - numer / denom

                    # update error
                    err = targets[:, itr] - np.dot(self.W_trained, self.Hx) # error is scalar

                    # update connectivity
                    self.W_trained = self.W_trained + err * np.dot(P, self.Hx)

        x_vals = np.transpose(x_vals)
        Hx_vals = np.transpose(Hx_vals)
        return x_vals, Hx_vals

    def run_rate(self, stim): 

        # initialize network activity to 0 
        # can be excluded to run from previous state
        self.x = np.zeros(self.N)
        self.Hx = np.tanh(self.x)

        # track variables
        x_vals = []
        Hx_vals = []

        t = 0
        itr = 0
        timesteps = int(self.T/self.dt)
        while itr < timesteps:
            # RK4 for each timestep
            self.rk4_step(stim, itr)

            t = t + self.dt
            itr = itr + 1

            # track variables
            x_vals.append(self.x)
            Hx_vals.append(self.Hx)
        
        x_vals = np.transpose(x_vals)
        Hx_vals = np.transpose(Hx_vals)
        return x_vals, Hx_vals