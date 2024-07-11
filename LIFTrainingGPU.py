import numpy as np
import scipy as sp
import cupy as cp
import matplotlib.pyplot as plt
from SpikeTraining import SpikeTraining
from LIFTraining import LIFTraining

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
            'stim_off': 0, # ms, matches run-forward time in FF_Demo
            'lam': 5, # learning rate factor
            'training_loops': 10, # number of training loops
            'train_every': 2, # ms, timestep of updating connectivity matrix
            'm': -57, # mean
            'std': (17), # standard deviation scalar, 1/sqrt(netsize)
            'cp': 1, # connection probability
            'runtime': 2000 # ms, runtime of trained network
        }
    return p

class LIFTrainingGPU(LIFTraining): 
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
        
        # output weighting
        self.W_out = np.zeros(self.N)

    def toGPU(self):
        # self.N = cp.asarray(self.N)
        # self.tau_s = cp.asarray(self.tau_s)
        # self.tau_f = cp.asarray(self.tau_f)
        # self.gain = cp.asarray(self.gain)
        # self.bias = cp.asarray(self.bias)
        # self.v_thr = cp.asarray(self.v_thr)
        # self.v_rest = cp.asarray(self.v_rest)
        # self.t_refract = cp.asarray(self.t_refract)

        self.slow = cp.asarray(self.slow)
        self.fast = cp.asarray(self.fast)
        self.refract = cp.asarray(self.refract) # time since last refractory period
        self.V = cp.asarray(self.V) # membrane voltages

        self.Jf = cp.asarray(self.Jf)
        self.Js = cp.asarray(self.Js)

        self.W_out = cp.asarray(self.W_out)

    def toCPU(self):
        # self.N = cp.asnumpy(self.N)
        # self.tau_s = cp.asnumpy(self.tau_s)
        # self.tau_f = cp.asnumpy(self.tau_f)
        # self.gain = cp.asnumpy(self.gain)
        # self.bias = cp.asnumpy(self.bias)
        # self.v_thr = cp.asnumpy(self.v_thr)
        # self.v_rest = cp.asnumpy(self.v_rest)
        # self.t_refract = cp.asnumpy(self.t_refract)

        self.slow = cp.asnumpy(self.slow)
        self.fast = cp.asnumpy(self.fast)
        self.refract = cp.asnumpy(self.refract) # time since last refractory period
        self.V = cp.asnumpy(self.V) # membrane voltages

        self.Jf = cp.asnumpy(self.Jf)
        self.Js = cp.asnumpy(self.Js)

        self.W_out = cp.asnumpy(self.W_out)
    
    def stepGPU(self, stim, itr): 
        ext = stim[:, itr] # can add check for in range
        # decay previous potentials
        ds = - self.dt/self.tau_s * self.slow
        df = - self.dt/self.tau_f * self.fast

        self.slow += ds
        self.fast += df

        # change in membrane potential
        dV = self.dt/self.tau_m * (self.v_rest - self.V
            + self.gain * (cp.dot(self.Js, self.slow) 
                           + cp.dot(self.Jf, self.fast) + ext) 
            + self.bias)
        
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

    def trainGPU_LIF(self, stim, targ, fout): # trains slow synaptic drive to match stim

        stimGPU = cp.asarray(stim)
        targGPU = cp.asarray(targ)
        foutGPU = cp.asarray(fout)

        self.toGPU() # move to GPU

        # initialize correlation matrix
        P = cp.eye(self.N, self.N) / self.lam
        timesteps = int(self.T/self.dt)

        for i in range(self.nloop):
            if i % int(self.nloop/5) == 0:
                print('training:', i)
                
            itr = 0
            while itr < timesteps:

                self.stepGPU(stimGPU, itr)
                if np.random.rand() < 1/(self.train_every * self.dt):
                    # train matrix
                    Ps = cp.dot(P, self.slow)

                    k = Ps / (1 + cp.dot(self.slow, Ps))
                    P = P - cp.outer(Ps, k)

                    err = cp.dot(self.Js, self.slow) - targGPU[:, itr]
                    oerr = cp.dot(self.W_out, self.slow) - foutGPU[itr] 

                    self.Js = self.Js - cp.outer(err, k)
                    self.W_out = self.W_out - oerr * k

                itr = itr + 1

        self.toCPU() # move to CPU
