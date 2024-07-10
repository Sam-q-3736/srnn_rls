import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from SpikeTraining import SpikeTraining

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
            'std': 1/np.sqrt(300), # standard deviation, 1/sqrt(netsize)
            'cp': 1, # connection probability
            'runtime': 2000 # ms, runtime of trained network
        }
    return p

class RateTraining(SpikeTraining): 
    def __init__(self, p):
        # initialize connectivity matrix
        self.W_init = self.genw_sparse(p['net_size'], p['m'], p['std'], p['cp'])
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

    # differential equation of dx/dt
    def dx(self, x, ext):
        return 1/self.tau_x * (-x + self.gain * np.dot(self.W_trained, np.tanh(x)) + ext)
        # return 1/self.tau_x * (-x + self.gain * np.dot(np.tanh(x), self.W_trained) + ext)

    def rk4_step(self, stim, itr):
        ext = stim[:, itr]

        x1 = self.dt * self.dx(self.x, ext)
        x2 = self.dt * self.dx(self.x + x1/2, ext)
        x3 = self.dt * self.dx(self.x + x2/2, ext)
        x4 = self.dt * self.dx(self.x + x3, ext)

        self.x = self.x + (x1 + 2*x2 + 2*x3 + x4) / 6
        self.Hx = np.tanh(self.x)

    def step(self, stim, itr):
        ext = stim[:, itr]

        self.x = self.x + self.dt * self.dx(self.x, ext)
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
        rel_errs = []

        t = 0
        itr = 0
        timesteps = int(self.T/self.dt)

        # training loop
        for i in range(self.nloop):
            if np.mod(i, 25) == 0:
                print('training trial', i)
            t = 0
            itr = 0
            
            while itr < timesteps: 
                
                # calculate next step of diffeqs
                self.rk4_step(stim, itr)

                # track variables
                x_vals.append(self.x)
                Hx_vals.append(self.Hx)

                # train connectivity matrix
                if itr > int(self.stim_off/self.dt) and itr < timesteps \
                    and np.random.rand() < 1/(self.train_every * self.dt):
                    # and np.mod(itr, int(self.train_every/self.dt)) == 0:

                    Phx = np.dot(P, self.Hx)

                    # update correlation matrix
                    numer = np.outer(Phx, Phx)
                    denom = 1 + np.dot(np.transpose(self.Hx), Phx)
                    P = P - numer / denom
                    # k = np.transpose(np.dot(P, self.Hx)) / denom
                    
                    # update error
                    err = np.dot(self.W_trained, self.Hx) - targets[:, itr] # error is vector
                    errs.append(np.linalg.norm(err))

                    # update connectivity
                    self.W_trained = self.W_trained - np.outer(err, np.dot(P, self.Hx))

                    #dws.append(np.linalg.norm(np.outer(err, np.dot(P, self.Hx))))
                    rel_errs.append(np.mean((np.dot(self.W_trained, self.Hx) - targets[:, itr]) / err))
                
                # update timestep
                t = t + self.dt
                itr = itr + 1

        x_vals = np.transpose(x_vals)
        Hx_vals = np.transpose(Hx_vals)
        return x_vals, Hx_vals, errs, rel_errs

    def run_rate(self, stim): 

        # initialize network activity to 0 
        # can be excluded to run from previous state
        # self.x = np.zeros(self.N)
        # self.Hx = np.tanh(self.x)

        # track variables
        x_vals = []
        Hx_vals = []

        #t = 0
        itr = 0
        timesteps = int(self.T/self.dt)
        while itr < timesteps:
            # RK4 for each timestep
            #self.rk4_step(stim, itr)
            self.step(stim, itr)

            # track variables
            x_vals.append(self.x)
            Hx_vals.append(self.Hx)

            #t = t + self.dt
            itr = itr + 1
        
        x_vals = np.transpose(x_vals)
        Hx_vals = np.transpose(Hx_vals)
        return x_vals, Hx_vals

    def fullFORCE(self, fin, fout):
        npar, tpar, trpar, cpar, rpar = create_default_params_rate()
        rpar['runtime'] = self.T
        DRNN = RateTraining(npar, tpar, trpar, cpar, rpar)
        Jd = DRNN.W_init

        # initialize random weighting inputs
        uind = sp.stats.uniform.rvs(size = npar['net_size']) * 2 - 1
        uin = sp.stats.uniform.rvs(size = npar['net_size']) * 2 - 1
        uout = sp.stats.uniform.rvs(size = npar['net_size']) * 2 - 1

        # generate inputs
        ufind = np.transpose(np.multiply(fin, uind))
        ufin = np.transpose(np.multiply(fin, uin))
        ufout = np.transpose(np.multiply(fout, uout))

        print('Stabilizing networks')
        for i in range(3): # 3 used in full-FORCE
            DRNN.run_rate(ufind + ufout)
            self.run_rate(ufin)

        # initialize variables
        timesteps = int(self.T/self.dt)
        P = np.eye(self.N) / self.lam
                
        # tracking variables
        x_vals = []
        Hx_vals = []
        errs = []
        rel_errs = []
        aux_targs = []

        # begin training
        itr = 0
        #t = 0
        print(self.nloop, 'total trainings')
        for i in range(self.nloop):
            if i % 20 == 0: print('training:', i)
            # _, dHx = DRNN.run_rate(ufin + ufout)
            # aux_targs = ufout + Jd @ dHx
            
            itr = 0
            #t = 0
            while itr < timesteps: 
                
                # calculate next step of diffeqs
                self.step(ufin, itr)
                DRNN.step(ufind + ufout, itr)
                aux_targ = np.dot(Jd, DRNN.Hx) + ufout[:, itr]

                # track variables
                x_vals.append(self.x)
                Hx_vals.append(self.Hx)
                aux_targs.append(aux_targ)

                # train connectivity matrix
                if np.random.rand() < 1/(self.train_every * self.dt):
                    # and np.mod(itr, int(self.train_every/self.dt)) == 0:

                    Phx = np.dot(P, self.Hx)
     
                    # update correlation matrix
                    # numer = np.outer(Phx, Phx)
                    # denom = 1 + np.dot(np.transpose(self.Hx), Phx)
                    k = Phx / (1 + np.dot(np.transpose(self.Hx), Phx))
                    P = P - np.outer(Phx, k)
                    
                    # update error
                    err = np.dot(self.W_trained, self.Hx) - aux_targ # error is vector
                    oerr = np.dot(self.W_out, self.Hx) - fout[itr] # error is scalar

                    # update connectivity
                    self.W_trained = self.W_trained - np.outer(err, k)

                    # update output weights
                    self.W_out = self.W_out - oerr * k
                    
                    # track training errors
                    errs.append(np.linalg.norm(err))
                    rel_errs.append(np.mean((np.dot(self.W_trained, self.Hx) - aux_targ) / err))
                
                # update timestep
                # t = t + self.dt
                itr = itr + 1

        x_vals = np.transpose(x_vals)
        Hx_vals = np.transpose(Hx_vals)
        aux_targs = np.transpose(aux_targs)
        return x_vals, Hx_vals, errs, rel_errs, aux_targs, ufin, ufout