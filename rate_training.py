import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from spike_training import *

def create_default_params():
    neuron_params = {
            'net_size': 300, # units in network
            'tau_x': 10, # ms, decay constant
            'gain': 1, # multiplier
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
    def dx(self, x, ext):
        return 1/self.tau_x * (-x + self.gain * np.dot(self.W_trained, self.Hx) + ext)
    
    def rk4_step(self, stim, itr):
        ext = stim[:, itr]

        x1 = self.dt * self.dx(self.x, ext)
        x2 = self.dt * self.dx(self.x + x1/2, ext)
        x3 = self.dt * self.dx(self.x + x2/2, ext)
        x4 = self.dt * self.dx(self.x + x3, ext)
        x_next = self.x + (x1 + 2*x2 + 2*x3 + x4) / 6

        self.x = x_next
        self.Hx = np.tanh(self.x)

    def train_rate(self, stim, targets):
        # initialize network activity to uniform random behavior
        # can be excluded to run from previous state
        # self.x = np.zeros(self.N)
        self.x = sp.stats.uniform.rvs(size = self.N) * 2 - 1
        self.Hx = np.tanh(self.x)

        # initialize correlation matrix
        P = np.eye(self.N) * 1/self.lam

        # track variables
        x_vals = []
        Hx_vals = []
        errs = []
        dws = []

        t = 0
        itr = 0
        timesteps = int(self.T/self.dt)

        # training loop
        for i in range(self.nloop):
            if np.mod(i, 50) == 0:
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
                    and np.random.rand() < 1/(self.train_every * self.dt):
                    # and np.mod(itr, int(self.train_every/self.dt)) == 0:
                    
                    # update correlation matrix
                    numer = np.outer(np.dot(P, self.Hx), np.dot(P, self.Hx))
                    denom = 1 + np.dot(self.Hx, np.dot(P, self.Hx))
                    P = P - numer / denom

                    # update error
                    err = targets[:, itr] - np.dot(self.W_trained, self.Hx) # error is vector
                    errs.append(np.linalg.norm(err))
                    # update connectivity
                    self.W_trained = self.W_trained + np.outer(err, np.dot(P, self.Hx))
                    dws.append(np.linalg.norm(np.outer(err, np.dot(P, self.Hx))))

        x_vals = np.transpose(x_vals)
        Hx_vals = np.transpose(Hx_vals)
        return x_vals, Hx_vals, errs, dws

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

    def fullFORCE(self, ufin, ufout):
        npar, tpar, trpar, cpar, rpar = create_default_params()
        DRNN = rate_training(npar, tpar, trpar, cpar, rpar)

        print('Training network...')
        P = np.eye(self.N)/self.lam
        Jd = DRNN.W_init
        J = self.W_trained

        dx = []
        x = []

        for i in range(self.nloop):
            t = 0
            itr = 0
            print('training', i)

            while itr < int(self.T/self.dt):
                DRNN.rk4_step(ufin + ufout, itr)
                rd = Jd @ DRNN.Hx
                
                self.rk4_step(ufin, itr)
                
                J = self.W_trained
                r = self.Hx

                dx.append(DRNN.Hx)
                x.append(self.Hx)

                if np.mod(itr, int(self.train_every/self.dt)):
                    J_err = (np.dot(J,r) - np.dot(Jd,rd) - ufout[:, itr])
                    Pr = np.dot(P,r)
                    k = np.transpose(Pr)/(1 + np.dot(np.transpose(r), Pr))
                    P = P - np.dot(Pr,k)

                    J = J - np.dot(J_err, k)
                    self.W_trained = J  
                
                itr = itr + 1

        return dx, x