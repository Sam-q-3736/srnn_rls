import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from SpikeTraining import SpikeTraining
from RateTraining import RateTraining

def create_default_params_rate():
    p = {
            'net_size': 300, # units in network
            'tau_x': 10, # ms, decay constant
            'gain': 1, # multiplier
            'total_time': 2000, # ms, total runtime
            'dt': 1, # ms
            'stim_on': 0, # ms
            'stim_off': 3, # ms, matches run-forward time in FF_Demo
            'lam': 1, # learning rate factor
            'training_loops': 10, # number of training loops
            'train_every': 2, # ms, timestep of updating connectivity matrix
            'runtime': 2000 # ms, runtime of trained network
        }
    return p

class fullFORCE(SpikeTraining): 
    def __init__(self, params):
        # initialize connectivity matrix
        self.p = params
        self.N = self.p['net_size']
        self.dt = self.p['dt']
        self.T = self.p['total_time']
        self.nloop = self.p['training_loops']
        self.train_every = self.p['train_every']
        self.tau_x = self.p['tau_x']

        self.W_init = self.genw_sparse(self.p['net_size'], 0, 1/self.p['net_size'], 1)
        self.W_trained = np.copy(self.W_init)
        self.W_out = np.zeros(self.N)

        # track current activity
        self.x = sp.stats.uniform.rvs(self.N) * 2 - 1
        self.Hx = np.tanh(self.x)

    def dx(self, x, ext):
        return 1/self.tau_x * (-x + np.dot(self.W_trained, np.tanh(x)) + ext)

    def step(self, stim, itr):
        ext = stim[:, itr]

        self.x = self.x + self.dt * self.dx(self.x, ext)
        self.Hx = np.tanh(self.x)

    def run_rate(self, stim): 
        itr = 0
        timesteps = int(self.T/self.dt)
        print(timesteps)
        while(itr < timesteps):
            self.step(stim, itr)
            itr += 1

    def ff_Train(self, fin, fout):

        DRNN = fullFORCE(self.p)
        Jd = DRNN.W_init

        # initialize random weighting inputs
        uind = sp.stats.uniform.rvs(size = self.N) * 2 - 1
        uin = sp.stats.uniform.rvs(size = self.N) * 2 - 1
        uout = sp.stats.uniform.rvs(size = self.N) * 2 - 1

        # generate inputs
        ufind = np.transpose(np.multiply(fin, uind))
        ufin = np.transpose(np.multiply(fin, uin))
        ufout = np.transpose(np.multiply(fout, uout))

        print("Stabilizing networks...")
        for i in range(3): # 3 used in full-FORCE
            DRNN.run_rate(ufind + ufout)
            self.run_rate(ufin)

        print('Training network...')
        P = np.eye(self.N)/self.p['lam']

        timesteps = int(self.T/self.dt)
        # begin training
        print(self.nloop, 'total trainings')

        for i in range(self.nloop):
            print('training:', i)

            itr = 0
            while itr < timesteps: 
                DRNN.step(ufind + ufout, itr)
                self.step(ufin, itr)

                # # train connectivity matrix
                # if np.random.rand() < 1/(self.train_every * self.dt):
                #     r = self.Hx
                #     rd = DRNN.Hx
                #     J = self.W_trained
                #     w = self.W_out

                #     J_err = np.dot(J, r) - np.dot(Jd, rd) - ufout[:, itr]
                #     w_err = np.dot(w, r) - fout[itr]

                #     Pr = np.dot(P, r)
                #     k = Pr / (1 + np.outer(r, Pr))
                #     P = P - np.outer(Pr, k)

                #     w = w - np.outer(w_err, k)
                #     J = J - np.outer(J_err, k)

                #     self.W_out = w
                #     self.W_trained = J
                itr += 1
