import numpy as np
import scipy as sp
import cupy as cp
import matplotlib.pyplot as plt
from SpikeTraining import SpikeTraining

def create_default_params_LIF():
    p = {
            'net_size': 300, # units in network
            'num_out': 1, # number of total outputs
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
            'lam': 5, # learning rate factor
            'training_loops': 10, # number of training loops
            'train_every': 2, # ms, timestep of updating connectivity matrix
            'm': -57, # mean
            'std': (17), # standard deviation scalar, 1/sqrt(netsize)
            'cp': 1, # connection probability
            'runtime': 1000 # ms, runtime of trained network
        }
    return p

class LIFTraining(SpikeTraining): 
    def __init__(self, p):
        # unpack parameters
        self.N = p['net_size']
        self.num_outs = p['num_out']
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

        self.lam = p['lam']
        self.nloop = p['training_loops']
        self.train_every = p['train_every']

        self.run_time = p['runtime']

        # initialize variables 
        self.slow = np.zeros(self.N) # slow synaptic connections
        self.fast = np.zeros(self.N) # fast synaptic connections
        self.refract = np.zeros(self.N) # time since last spike
        self.V = np.zeros(self.N) + self.v_rest # membrane voltages

        # fast and slow connectivity 
        self.Jf = self.genw_sparse(p['net_size'], p['m']/p['net_size'], p['std']/np.sqrt(p['net_size']), p['cp'])
        self.Js = np.zeros((self.N, self.N))
        
        # output weighting(s)
        self.W_out = self.W_out = np.zeros((self.num_outs, self.N))
    
    def toGPU(self):
        # move all matrices and vectors to the GPU
        self.slow = cp.asarray(self.slow)
        self.fast = cp.asarray(self.fast)
        self.refract = cp.asarray(self.refract) 
        self.V = cp.asarray(self.V) 

        self.Jf = cp.asarray(self.Jf)
        self.Js = cp.asarray(self.Js)

        self.W_out = cp.asarray(self.W_out)

    def toCPU(self):
        # move GPU matrices and vectors to the CPU
        self.slow = cp.asnumpy(self.slow)
        self.fast = cp.asnumpy(self.fast)
        self.refract = cp.asnumpy(self.refract) 
        self.V = cp.asnumpy(self.V) 

        self.Jf = cp.asnumpy(self.Jf)
        self.Js = cp.asnumpy(self.Js)

        self.W_out = cp.asnumpy(self.W_out)

    def step(self, stim: np.ndarray, itr: int): 

        ext = stim[:, itr]

        # decay previous potentials
        dslow = np.multiply(self.slow, -self.dt/self.tau_s)
        dfast = np.multiply(self.fast, -self.dt/self.tau_f)

        self.slow = np.add(self.slow, dslow)
        self.fast = np.add(self.fast, dfast)

        # change in membrane potential
        dV = self.dt/self.tau_m * (self.v_rest - self.V
            + self.gain * (np.dot(self.Js, self.slow)
                           + np.dot(self.Jf, self.fast) + ext) 
            + self.bias)
        
        self.V += dV

        # enforce refractory period
        idxrefract = self.refract > 0 # get neurons in refractory period
        self.V[idxrefract] = self.v_rest # hold at rest voltage
        self.refract[idxrefract] -= 1 # decrease refractory period

        idxspike = self.V > self.v_thr # get neurons which spike
        self.refract[idxspike] = int(self.t_refract/self.dt) # set refractory periods
        self.slow[idxspike] += 1 
        self.fast[idxspike] += 1

    def stepGPU(self, stim: cp.ndarray, itr: int): 
        
        ext = stim[:, itr]

        # decay previous potentials
        dslow = cp.multiply(self.slow, -self.dt/self.tau_s)
        dfast = cp.multiply(self.fast, -self.dt/self.tau_f)

        self.slow = cp.add(self.slow, dslow)
        self.fast = cp.add(self.fast, dfast)

        # change in membrane potential
        dV = self.dt/self.tau_m * (self.v_rest - self.V
            + self.gain * (cp.dot(self.Js, self.slow)
                           + cp.dot(self.Jf, self.fast) + ext) 
            + self.bias)
        
        self.V += dV

        # enforce refractory period
        idxrefract = self.refract > 0 # get neurons in refractory period
        self.V[idxrefract] = self.v_rest # hold at rest voltage
        self.refract[idxrefract] -= 1 # decrease refractory period

        idxspike = self.V > self.v_thr # get neurons which spike
        self.refract[idxspike] = int(self.t_refract/self.dt) # set refractory periods
        self.slow[idxspike] += 1 
        self.fast[idxspike] += 1

    def run(self, stim: np.ndarray): 

        timesteps = int(self.run_time / self.dt)

        # tracking variables
        voltage = np.zeros((timesteps, self.N))
        slow_curr = np.zeros((timesteps, self.N))
        fast_curr = np.zeros((timesteps, self.N))

        for itr in range(timesteps):
            
            self.step(stim, itr)
            voltage[itr, :] = np.copy(self.V)
            slow_curr[itr, :] = np.copy(self.slow)
            fast_curr[itr, :] = np.copy(self.fast)
            
        voltage = np.transpose(voltage)
        slow_curr = np.transpose(slow_curr)
        fast_curr = np.transpose(fast_curr)

        return voltage, slow_curr, fast_curr

    def runGPU(self, stim: np.ndarray): 

        timesteps = int(self.run_time / self.dt)

        stimGPU = cp.asarray(stim)
        self.toGPU()

        # tracking variables
        voltage = cp.zeros((timesteps, self.N))
        slow_curr = cp.zeros((timesteps, self.N))
        fast_curr = cp.zeros((timesteps, self.N))

        for itr in range(timesteps):
            
            self.stepGPU(stimGPU, itr)
            voltage[itr, :] = np.copy(self.V)
            slow_curr[itr, :] = np.copy(self.slow)
            fast_curr[itr, :] = np.copy(self.fast)

        voltage = cp.transpose(voltage)
        slow_curr = cp.transpose(slow_curr)
        fast_curr = cp.transpose(fast_curr)

        self.toCPU()
        voltage = cp.asnumpy(voltage)
        slow_curr = cp.asnumpy(slow_curr)
        fast_curr = cp.asnumpy(fast_curr)
            
        return voltage, slow_curr, fast_curr

    def train(self, stim: np.ndarray, targ: np.ndarray, fout: np.ndarray): 

        timesteps = int(self.T/self.dt)

        # initialize correlation matrix
        P = np.eye(self.N, self.N) / self.lam

        for i in range(self.nloop):

            for itr in range(timesteps):

                self.step(stim, itr)
                if np.random.rand() < 1/(self.train_every * self.dt):

                    Ps = np.dot(P, self.slow)

                    k = Ps / (1 + np.dot(self.slow, Ps))
                    P = P - np.outer(Ps, k)

                    err = np.dot(self.Js, self.slow) - targ[:, itr]
                    oerr = np.dot(self.W_out, self.slow) - fout[:, itr] 

                    self.Js = self.Js - np.outer(err, k)
                    self.W_out = self.W_out - np.outer(oerr, k)

    def trainGPU(self, stim: np.ndarray, targ: np.ndarray, fout: np.ndarray): 

        timesteps = int(self.T/self.dt)

        stimGPU = cp.asarray(stim)
        targGPU = cp.asarray(targ)
        foutGPU = cp.asarray(fout)
        self.toGPU()

        # initialize correlation matrix
        P = cp.eye(self.N, self.N) / self.lam

        for i in range(self.nloop):

            for itr in range(timesteps):

                self.stepGPU(stimGPU, itr)
                if np.random.rand() < 1/(self.train_every * self.dt):

                    Ps = cp.dot(P, self.slow)

                    k = Ps / (1 + cp.dot(self.slow, Ps))
                    P = P - cp.outer(Ps, k)

                    err = cp.dot(self.Js, self.slow) - targGPU[:, itr]
                    oerr = cp.dot(self.W_out, self.slow) - foutGPU[:, itr] 

                    self.Js = self.Js - cp.outer(err, k)
                    self.W_out = self.W_out - cp.outer(oerr, k)
        
        self.toCPU()

    def depasquale(self, fin, fout):
        from RateTraining import create_default_params_rate
        from RateTraining import RateTraining

        p = create_default_params_rate()
        p['runtime'] = self.T
        DRNN = RateTraining(p)
        Jd = DRNN.W_trained

        # initialize random weighting inputs
        uind = np.random.rand(self.num_outs, self.N) * 2 - 1
        uin = np.random.rand(1, self.N) * 2 - 1

        # generate inputs
        ufind = np.transpose(uind[0] * fin)
        ufout = np.transpose(uind[1] * fout)

        ufin = np.transpose(uin * fin)
        dinp = ufind + ufout

        print('Stabilizing networks')
        for i in range(3): # 3 used in full-FORCE
            DRNN.run(dinp)
            self.run(ufin)

        # initialize variables
        timesteps = int(self.T/self.dt)
        P = np.eye(self.N) / self.lam
                
        # tracking variables
        voltage = np.zeros((self.nloop * timesteps, self.N))
        slow = np.zeros((self.nloop * timesteps, self.N))
        fast = np.zeros((self.nloop * timesteps, self.N))
        # errs = []
        # rel_errs = []
        aux_targs = np.zeros((self.nloop * timesteps, self.N))

        # begin training
        print(self.nloop, 'total trainings')
        for i in range(self.nloop):
            #if i % 20 == 0: print('training:', i)
            
            for itr in range(timesteps): 
                
                # calculate next step of diffeqs
                self.step(ufin, itr)
                DRNN.step(dinp, itr)

                # track variables
                voltage[itr + i * timesteps, :] = self.V
                slow[itr + i * timesteps, :] = self.slow
                fast[itr + i * timesteps, :] = self.fast
                aux_targs[itr + i * timesteps, :] = np.dot(Jd, DRNN.Hx) + ufout[:, itr]

                # train connectivity matrix
                if np.random.rand() < 1/(self.train_every * self.dt):

                    Phx = np.dot(P, self.slow)
     
                    # update correlation matrix
                    k = Phx / (1 + np.dot(np.transpose(self.slow), Phx))
                    P = P - np.outer(Phx, k)
                    
                    # update error
                    err = np.dot(self.Js, self.slow) - aux_targs[itr + i * timesteps, :] # error is vector
                    oerr = np.dot(self.W_out, self.slow) - fout[itr, :] # error is scalar

                    # update connectivity
                    self.Js = self.Js - np.outer(err, k)

                    # update output weights
                    self.W_out = self.W_out - np.outer(oerr, k)
                    
                    # track training errors
                    # errs.append(np.linalg.norm(err))
                    # rel_errs.append(np.mean((np.dot(self.W_trained, self.Hx) - aux_targs[itr + i * timesteps, :]) / err))
                
        slow = np.transpose(slow)
        fast = np.transpose(fast)
        aux_targs = np.transpose(aux_targs)
        return voltage, slow, fast, aux_targs, ufin, ufout