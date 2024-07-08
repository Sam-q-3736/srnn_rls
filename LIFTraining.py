import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from SpikeTraining import SpikeTraining

def create_default_params_LIF():
    p = {
            'net_size': 300, # units in network
            'tau_s': 100, # ms, slow decay constant
            'tau_f': 2, # ms, fast decay constant
            'tau_m': 20, # ms, membrane decay constant
            'gain': 7, # mV, multiplier of synaptic drive
            'bias': 10, # mV, bias current
            'v_thr': -55, # mV, threshold
            'v_rest': -65, # mV, resting voltage
            't_refract': 2, # ms, refractory period 
            'total_time': 1000, # ms, total runtime
            'dt': 1, # ms
            'stim_on': 0, # ms
            'stim_off': 50, # ms, matches run-forward time in FF_Demo
            'lam': 5, # learning rate factor
            'training_loops': 10, # number of training loops
            'train_every': 2, # ms, timestep of updating connectivity matrix
            'm': -57, # mean
            'std': (17), # standard deviation scalar, 1/sqrt(netsize)
            'cp': 1, # connection probability
            'runtime': 2000 # ms, runtime of trained network
        }
    return p

class LIFTraining(SpikeTraining): 
    def __init__(self, p):
        # unpack parameters
        self.N = p['net_size']
        self.tau_s = p['tau_s']
        self.tau_f = p['tau_f']
        self.tau_m = p['tau_m']
        self.gain = p['gain']
        self.bias = p['bias']
        self.v_thr = p['v_thr']
        self.v_rest = p['v_rest']
        self.t_refract = p['t_refract']

        self.T = p['total_time']
        self.dt = p['dt']

        self.stim_on = p['stim_on']
        self.stim_off = p['stim_off']
        
        self.lam = p['lam']
        self.nloop = p['training_loops']
        self.train_every = p['train_every']

        self.run_time = p['runtime']

        # initialize variables 
        self.slow = np.zeros(self.N)
        self.fast = np.zeros(self.N)
        self.refract = np.zeros(self.N) # time since last refractory period
        self.V = np.zeros(self.N) + self.v_rest # membrane voltages

        # fast and slow connectivity 
        self.Jf = self.genw_sparse(p['net_size'], p['m']/p['net_size'], p['std']/np.sqrt(p['net_size']), p['cp'])
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
        ds = - self.dt/self.tau_s * self.slow
        df = - self.dt/self.tau_f * self.fast

        self.slow += ds
        self.fast += df

        # change in membrane potential
        dV = self.dt * self.dV(ext)
        self.V += dV

        idxr = self.refract > 0 # get neurons in refractory period
        self.V[idxr] = self.v_rest # hold at rest voltage
        self.refract[idxr] -= 1 # decrease refractory period

        idxs = self.V > self.v_thr # get neurons which spike
        self.refract[idxs] = int(self.t_refract/self.dt) # set refractory periods
        self.slow[idxs] += 1 
        self.fast[idxs] += 1

    def run_LIF(self, stim): 
        
        # # initialize variables to base states
        # self.slow = np.zeros(self.N)
        # self.fast = np.zeros(self.N)
        # self.refract = np.zeros(self.N)
        # self.V = np.zeros(self.N) + self.v_rest

        itr = 0
        timesteps = int(self.run_time / self.dt)

        # tracking variables
        voltage = np.zeros((timesteps, self.N))
        slow_curr = np.zeros((timesteps, self.N))
        fast_curr = np.zeros((timesteps, self.N))

        while(itr < timesteps):
            
            self.step(stim, itr)
            voltage[itr] = np.copy(self.V)
            slow_curr[itr] = np.copy(self.slow)
            fast_curr[itr] = np.copy(self.fast)
            
            itr += 1

        return np.transpose(voltage), np.transpose(slow_curr), np.transpose(fast_curr)

    def train_LIF(self, stim, targ): # trains slow synaptic drive to match stim

        # initialize correlation matrix
        P = np.eye(self.N, self.N) / self.lam
        timesteps = int(self.T/self.dt)

        for i in range(self.nloop):
            if i % 20 == 0:
                print('training:', i)
            itr = 0
            while itr < timesteps:

                self.step(stim, itr)
                if np.random.rand() < 1/(self.train_every * self.dt):
                    # train matrix
                    Ps = np.dot(P, self.slow)

                    k = Ps / (1 + np.dot(self.slow, Ps))
                    P = P - np.outer(Ps, k)

                    err = np.dot(self.Js, self.slow) - targ[:, itr]

                    self.Js = self.Js - np.outer(err, k)

                    # Ps = np.dot(P, self.slow)[:, np.newaxis]

                    # k = np.transpose(Ps) \
                    #     / (1 + np.dot(np.transpose(self.slow[:, np.newaxis]), Ps))
                    
                    # P = P - np.dot(Ps, k)

                    # err = np.dot(self.Js, self.slow) - targ[:, itr]

                    # self.Js = self.Js - np.dot(err[:, np.newaxis], k)

                itr = itr + 1
