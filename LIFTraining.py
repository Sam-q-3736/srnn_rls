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
            'total_time': 1000, # ms, total runtime
            'dt': 0.1, # ms
            'stim_on': 0, # ms
            'stim_off': 3 # ms, matches run-forward time in FF_Demo
        }    
    train_params = {
            'lam': 1, # learning rate factor
            'training_loops': 10, # number of training loops
            'train_every': 2 # ms, timestep of updating connectivity matrix
        }
    connectivity_params = {
            'm': -57/neuron_params['net_size'], # mean
            'std': (17 ** 2)/np.sqrt(neuron_params['net_size']), # standard deviation, 1/sqrt(netsize)
            'cp': 1, # connection probability
        }
    run_params = {
            'runtime': 2000 # ms, runtime of trained network
        }
    return neuron_params, time_params, train_params, connectivity_params, run_params

class LIFTraining(SpikeTraining): 
    def __init__(self, neuron_params, time_params, train_params, connectivity_params, run_params):
        # unpack parameters
        self.N = neuron_params['net_size']
        self.tau_s = neuron_params['tau_s']
        self.tau_f = neuron_params['tau_f']
        self.gain = neuron_params['gain']
        self.bias = neuron_params['bias']
        self.v_thr = neuron_params['v_thr']
        self.v_rest = neuron_params['v_rest']
        self.t_refract = neuron_params['t_refract']

        self.T = time_params['total_time']
        self.dt = time_params['dt']

        self.stim_on = time_params['stim_on']
        self.stim_off = time_params['stim_off']
        
        self.lam = train_params['lam']
        self.nloop = train_params['training_loops']
        self.train_every = train_params['train_every']

        self.run_time = run_params['runtime']

        # initialize variables 
        self.slow = np.zeros(self.N)
        self.fast = np.zeros(self.N)
        self.refract = np.zeros(self.N) # time since last refractory period
        self.V = np.zeros(self.N) - self.v_rest # membrane voltages

        # fast and slow connectivity 
        self.Jf = self.genw_sparse(neuron_params['net_size'], connectivity_params['m'], connectivity_params['std'], connectivity_params['cp'])
        self.Js = np.zeros((self.N, self.N))

    def dslow(self): 
        return -1/self.tau_s * self.slow

    def dfast(self):
        return -1/self.tau_f * self.fast
    
    def dV(self, ext):
        return 1/self.tau_m * (self.v_rest - self.V
            + self.gain * (self.Js @ self.slow 
                           + self.Jf @ self.fast + ext) 
            + self.bias)
    
    def step(self, stim, itr): 
        ext = stim[:, itr]
        # decay previous potentials
        ds = self.dt * self.dslow()
        df = self.dt * self.dfast()

        # change in membrane potential
        dV = self.dt * self.dV(ext)
        idxr = self.refract > 0 # get neurons in refractory period
        dV[idxr] = self.v_rest # hold at rest voltage
        self.refract[idxr] -= 1 # decrease refractory period

        idxs = dV > self.v_thr # get neurons which spike
        self.refract[idxs] = int(self.t_refract/self.dt) # set refractory periods
        ds[idxs] += 1 
        df[idxs] += 1

    def run_LIF(self, stim): 
        pass