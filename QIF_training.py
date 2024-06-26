import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from spike_training import *

def create_default_params():
        neuron_params = {
            'net_size': 200, # units in network
            'tau': 1, # ms, neuron decay constant
            'tau_s': 20, # ms, synaptic filtering constant
        }
        time_params = {
            'total_time': 1000, # ms, total runtime 
            'dt': 0.1, # ms, timestep for differential equations
            'stim_on': 0, # ms, start of stimulus on
            'stim_off': 50 # ms, stim off time
        }
        train_params = {
            'lam': 1, # learning rate factor
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

class QIF_training(spike_training):

    def __init__(self, neuron_params, time_params, train_params, connectivity_params, run_params):
        # initialize connectivity matrix
        self.W_init = self.genw_sparse(neuron_params['net_size'], connectivity_params['m'], connectivity_params['std'], connectivity_params['cp'])
        self.W_trained = np.copy(self.W_init)

        # unpack parameters
        self.N = neuron_params['net_size']
        self.tau = neuron_params['tau']
        self.tau_s = neuron_params['tau_s']

        self.T = time_params['total_time']
        self.dt = time_params['dt']

        self.stim_on = time_params['stim_on']
        self.stim_off = time_params['stim_off']
        
        self.lam = train_params['lam']
        self.nloop = train_params['training_loops']
        self.train_every = train_params['train_every']

        self.run_time = run_params['runtime']

        # initialize variables
        self.theta = np.zeros(self.N) # phase of neurons
        self.u = np.zeros(self.N) # synaptic drive
        self.r = np.zeros(self.N) # filtered synaptic drive
        self.spk_t = np.zeros(self.N) # discrete spikes

    def dtheta(self, theta_in, n_drive):
        return 1/self.tau * (1 - np.cos(theta_in) + np.multiply(n_drive, (1 + np.cos(theta_in))))

    def dr(self, r_in): 
        return -1/self.tau_s * r_in
        # does not include addition of new spikes
        
    def rk4_step(self, stim, itr): 
        ext = stim[:, itr]
        
        # RK4 for theta
        k1 = self.dt * self.dtheta(self.theta, self.u + ext);
        k2 = self.dt * self.dtheta(self.theta + k1/2, self.u + ext);
        k3 = self.dt * self.dtheta(self.theta + k2/2, self.u + ext);
        k4 = self.dt * self.dtheta(self.theta + k3, self.u + ext);
        theta_next = self.theta + (k1 + 2*k2 + 2*k3 + k4)/6;

        # RK4 for filtered spike train (one step)
        r1 = self.dt * self.dr(self.r) 
        r2 = self.dt * self.dr(self.r + r1/2)
        r3 = self.dt * self.dr(self.r + r2/2)
        r4 = self.dt * self.dr(self.r + r3)
        r_next = self.r + (r1 + 2*r2 + 2*r3 + r4)/6

        # spike detection
        idx1 = theta_next - self.theta > 0
        idx2 = theta_next - self.theta > np.mod(np.pi - self.theta, 2*np.pi) # surely there's a better way to do this...
        idx = np.multiply(idx1, idx2)
        self.r[idx] += 1/self.tau_s # update spikes in r
        self.spk_t[idx] += 1

        # update variables
        self.theta = np.mod(self.theta + (k1 + 2*k2 + 2*k3 + k4)/6, 2*np.pi)
        self.r = self.r + (r1 + 2*r2 + 2*r3 + r4)/6
        self.u = np.dot(self.W_trained, self.r)

    def train_QIF(self, stim, targets):
            
        # initialize variables
        # exclude to run from previous behavior
        self.theta = np.zeros(self.N) # phase of neurons
        self.u = np.zeros(self.N) # synaptic drive
        self.r = np.zeros(self.N) # filtered synaptic drive
        self.spk_t = np.zeros(self.N) # tracks spikes
        
        # initialize correlation matrices for each row
        Pidx = np.zeros((self.N, self.N))
        Ps = []
        for row in range(self.N):
            Pidx[row] = (self.W_trained[row] != 0)
            tot = int(np.sum(Pidx[row]))
            Ps.append(1/self.lam * np.identity(tot))
        Pidx = Pidx.astype(bool)

        # optional variables to track states
        spks = []
        sdrive = []
        thetas = []
        spk_rast = []
        
        # training loop
        for i in range(self.nloop):
            print('training trial', i)
            t = 0
            itr = 0
            timesteps = int(self.T/self.dt)
            while itr < timesteps:

                # calculate next timestep of variables and update
                self.rk4_step(stim, itr) 

                # update timestep
                t = t + self.dt
                itr = itr + 1 

                # track variables 
                spks.append(self.r)
                sdrive.append(self.u)  
                thetas.append(self.theta)
                spk_rast.append(self.spk_t)

                # reset current spikes to 0
                self.spk_t = np.zeros(self.N)

                # train W matrix
                if t > self.stim_off and t < int(self.T) \
                    and np.mod(itr, int(self.train_every/self.dt)) == 0: # only train after initial stimulus
                    
                    for row in range(self.N): # update each row of W by RLS

                        # update correlation matrix
                        numer = np.outer(np.dot(Ps[row], self.r[Pidx[row]]), np.dot(Ps[row], self.r[Pidx[row]]))
                        denom = 1 + np.dot(self.r[Pidx[row]], np.dot(Ps[row], self.r[Pidx[row]]))
                        Ps[row] = Ps[row] - numer / denom

                        # update error term
                        err = targets[row, itr] - \
                            np.dot(self.W_trained[row, Pidx[row]], self.r[Pidx[row]]) # error is scalar

                        # update W
                        self.W_trained[row, Pidx[row]] \
                            = self.W_trained[row, Pidx[row]] + err * np.dot(Ps[row], self.r[Pidx[row]])
            
        spks = np.transpose(spks)
        sdrive = np.transpose(sdrive)
        thetas = np.transpose(thetas)
        spk_rast = np.transpose(spk_rast)
        return spks, sdrive, thetas, spk_rast

    def run_QIF(self, stim):
            
        # initialize variables
        # exclude to run from previous behavior
        self.theta = np.zeros(self.N) # phase of neurons
        self.u = np.zeros(self.N) # synaptic drive
        self.r = np.zeros(self.N) # filtered synaptic drive
        self.spk_t = np.zeros(self.N) # tracks spikes

        # optional variables to track states
        spks = []
        sdrive = []
        thetas = []
        spk_rast = []
        
        # training loop
        t = 0.0
        itr = 0
        timesteps = int(self.T/self.dt)
        while itr < timesteps:

            # calculate next timestep of variables and update
            self.rk4_step(stim, itr) 

            # update timestep
            t = t + self.dt
            itr = itr + 1

            # track variables 
            spks.append(self.r)
            sdrive.append(self.u)  
            thetas.append(self.theta)
            spk_rast.append(self.spk_t)

            # reset current spikes to 0
            self.spk_t = np.zeros(self.N)

        spks = np.transpose(spks)
        sdrive = np.transpose(sdrive)
        thetas = np.transpose(thetas)
        spk_rast = np.transpose(spk_rast)
        return spks, sdrive, thetas, spk_rast