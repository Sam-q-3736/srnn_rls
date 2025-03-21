import numpy as np
import scipy as sp
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from SpikeTraining import SpikeTraining

def create_default_params_rate():
    p = {
            'net_size': 300, # units in network
            'num_out': 1, # number of outputs
            'tau_x': 10, # ms, decay constant
            'gain': 1, # multiplier
            'total_time': 2000, # ms, total runtime
            'dt': 1, # ms
            'lam': 1, # learning rate factor
            'training_loops': 20, # number of training loops
            'train_every': 2, # ms, timestep of updating connectivity matrix
            'm': 0, # mean
            'std': 1, # standard deviation multiplier, 1/sqrt(netsize)
            'cp': 1, # connection probability
            'runtime': 2000 # ms, runtime of trained network
        }
    return p

class RateTraining(SpikeTraining): 
    def __init__(self, p):
        # unpack parameters
        self.N = p['net_size']
        self.num_outs = p['num_out']
        self.tau_x = p['tau_x']
        self.gain = p['gain']

        self.T = p['total_time']
        self.dt = p['dt']

        self.lam = p['lam']
        self.nloop = p['training_loops']
        self.train_every = p['train_every']

        self.run_time = p['runtime']

        # initialize connectivity matrix
        self.W_trained = self.genw_sparse(self.N, p['m'], p['std']/np.sqrt(self.N), p['cp'])

        # initalize output weights
        self.W_out = np.zeros((self.num_outs, p['net_size']))

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

    def step(self, stim: np.ndarray, itr: int):
        ext = stim[:, itr]
        self.x = self.x + self.dt/self.tau_x * (-self.x + self.gain * np.dot(self.W_trained, np.tanh(self.x)) + ext)
        self.Hx = np.tanh(self.x)

    def stepGPU(self, stim: cp.ndarray, itr: int):
        ext = stim[:, itr]
        self.x = self.x + self.dt * 1/self.tau_x * (-self.x + self.gain * cp.dot(self.W_trained, cp.tanh(self.x)) + ext)
        self.Hx = cp.tanh(self.x)

    def run(self, stim: np.ndarray): 
        timesteps = int(self.T/self.dt)

        # track variables
        x_vals = np.zeros((timesteps, self.N))
        Hx_vals = np.zeros((timesteps, self.N))

        for itr in range(timesteps):

            self.step(stim, itr)

            # track variables
            x_vals[itr, :] = (self.x)
            Hx_vals[itr, :] = (self.Hx)
        
        x_vals = np.transpose(x_vals)
        Hx_vals = np.transpose(Hx_vals)
        return x_vals, Hx_vals

    def runGPU(self, stim: np.ndarray): 
        itr = 0
        timesteps = int(self.T/self.dt)
        
        # track variables
        x_vals = cp.zeros((timesteps, self.N))
        Hx_vals = cp.zeros((timesteps, self.N))

        # move variables to GPU
        stimGPU = cp.asarray(stim)
        self.toGPU()

        for itr in range(timesteps):
            self.stepGPU(stimGPU, itr)

            # track variables
            x_vals[itr, :] = self.x
            Hx_vals[itr, :] = self.Hx
                
        x_vals = cp.transpose(x_vals)
        Hx_vals = cp.transpose(Hx_vals)

        # move variables to CPU
        self.toCPU()
        x_vals = cp.asnumpy(x_vals)
        Hx_vals = cp.asnumpy(Hx_vals)

        return x_vals, Hx_vals

    def train(self, stim: np.ndarray, targ: np.ndarray, fout: np.ndarray):
        # initialize correlation matrix
        P = np.eye(self.N) * 1/self.lam
        timesteps = int(self.T/self.dt)

        # training loop
        for i in range(self.nloop):
            
            for itr in range(timesteps):
                
                # calculate next step of diffeqs
                self.step(stim, itr)

                # train connectivity matrix
                if itr < timesteps and np.random.rand() < 1/(self.train_every * self.dt):

                    Phx = np.dot(P, self.Hx)
                    k = Phx / (1 + np.dot(self.Hx, Phx))
                    P = P - np.outer(Phx, k)
                    
                    # update error
                    err = np.dot(self.W_trained, self.Hx) - targ[:, itr]
                    oerr = np.dot(self.W_out, self.Hx) - fout[:, itr]

                    # update connectivity
                    self.W_trained = self.W_trained - np.outer(err, k)
                    self.W_out = self.W_out - np.outer(oerr, k)
                
    def trainGPU(self, stim: np.ndarray, targ: np.ndarray, fout: np.ndarray):
        # initialize correlation matrix
        P = cp.eye(self.N, self.N) / self.lam
        timesteps = int(self.T/self.dt)

        stimGPU = cp.asarray(stim)
        targGPU = cp.asarray(targ)
        foutGPU = cp.asarray(fout)
        self.toGPU() # move to GPU

        for i in range(self.nloop):
            
            for itr in range(timesteps):
                
                # calculate next step of diffeqs
                self.stepGPU(stimGPU, itr)

                # train connectivity matrix
                if itr < timesteps and np.random.rand() < 1/(self.train_every * self.dt):

                    Phx = cp.dot(P, self.Hx)
                    k = Phx / (1 + cp.dot(self.Hx, Phx))
                    P = P - cp.outer(Phx, k)
                    
                    # update error
                    err = cp.dot(self.W_trained, self.Hx) - targGPU[:, itr]
                    oerr = cp.dot(self.W_out, self.Hx) - foutGPU[:, itr]

                    # update connectivity
                    self.W_trained = self.W_trained - cp.outer(err, k)
                    self.W_out = self.W_out - cp.outer(oerr, k)
                        
        self.toCPU() # move to CPU

    def ff(self, fin, fout): 
        
        # First, initialize some parameters
        N = self.N
        self.W_trained = np.zeros((N,N))
        self.W_out = np.zeros((self.num_outs, N))
  
		# Then instantiate the network and pull out some relevant weights
        p = create_default_params_rate()
        DRNN = RateTraining(p)
		
        ufind = np.outer(sp.stats.uniform.rvs(size = self.N), fin)
        ufout = np.outer(sp.stats.uniform.rvs(size = self.N), fout)

        ufin = np.outer(sp.stats.uniform.rvs(size = self.N), fin)

        # w_targ = np.transpose(DRNN.rnn_par['inp_weights'][D_num_inputs:(D_num_inputs+D_num_targs),:])
		# w_hint = np.transpose(DRNN.rnn_par['inp_weights'][(D_num_inputs+D_num_targs):D_num_total_inps,:])
        Jd = DRNN.W_trained

		################### Monitor training with these variables:
        J_err_ratio = []
        J_err_mag = []
        J_norm = []

        w_err_ratio = []
        w_err_mag = []
        w_norm = []
		###################

		# Let the networks settle from the initial conditions
        print('Initializing',end="")
        for i in range(3):
            print('.',end="")
			# inp, targ, hints = inps_and_targs(dt=p['dt'], **kwargs)[0:3]
			# D_total_inp = np.hstack((inp,targ,hints))
            DRNN.run(ufind + ufout)
            self.run(ufind)
        print('')

		# Now begin training
        print('Training network...')
		# Initialize the inverse correlation matrix
        P = np.eye(N)/self.lam
        for trial in range(self.nloop):
            if np.mod(trial,50)==0:
                print('')
            print('.',end="")

            # For recording:
            # dx = [] # Driven network activity
            # x = []	# RNN activity
            # z = []	# RNN output
            for itr in range(self.T):
                # Run both RNNs forward and get the activity. Record activity for potential plotting
                DRNN.step(ufind + ufout, itr)
                self.step(ufin, itr)

                # dx.append(np.squeeze(np.tanh(dx_t) + np.arange(5)*2))
                # z.append(np.squeeze(z_t))
                # x.append(np.squeeze(np.tanh(x_t[:,0:5]) + np.arange(5)*2))

                if np.random.rand() < (1/self.train_every):
                    # Extract relevant values
                    r = self.Hx # column vector
                    rd = DRNN.Hx # column vector
                    J = self.W_trained
                    w = self.W_out

                    # Now for the RLS algorithm:
                    # Compute errors
                    J_err = (np.dot(J,r) - np.dot(Jd,rd) 
                            - ufout[:, itr])

                    w_err = np.dot(w,r) - fout[itr]

                    # Compute the gain (k) and running estimate of the inverse correlation matrix
                    Pr = np.dot(P,r)
                    k = Pr / (1 + np.dot(self.Hx, Pr))
                    P = P - np.outer(Pr, k)

                    # Update weights
                    w = w - np.outer(w_err, k)
                    J = J - np.outer(J_err, k)

                    self.W_trained = J
                    self.W_out = w

                    # if monitor_training==1:
                    #     J_err_plus = (np.dot(J,r) - np.dot(Jd,rd) 
                    #                 -np.dot(w_targ,targ[t:(t+1),:].T) - np.dot(w_hint, hints[t:(t+1),:].T))
                    #     J_err_ratio = np.hstack((J_err_ratio, np.squeeze(np.mean(J_err_plus/J_err))))
                    #     J_err_mag = np.hstack((J_err_mag, np.squeeze(np.linalg.norm(J_err))))
                    #     J_norm = np.hstack((J_norm,np.squeeze(np.linalg.norm(J))))

                    #     w_err_plus = np.dot(w,r) - targ[t:(t+1),:].T
                    #     w_err_ratio = np.hstack((w_err_ratio, np.squeeze(w_err_plus/w_err)))
                    #     w_err_mag = np.hstack((w_err_mag, np.squeeze(np.linalg.norm(w_err))))
                    #     w_norm = np.hstack((w_norm, np.squeeze(np.linalg.norm(w))))
        return ufin
    
    # to be refactored
    def fullFORCE(self, fin, fout):
        p = create_default_params_rate()
        p['runtime'] = self.T
        DRNN = RateTraining(p)
        Jd = DRNN.W_trained

        self.W_trained = np.zeros((self.N, self.N))

        # initialize random weighting inputs
        # np.random.seed(0)
        uind = np.random.rand(3, p['net_size']) * 2 - 1

        # np.random.seed(123456789)
        uin = np.random.rand(1, p['net_size']) * 2 - 1

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
        x_vals = np.zeros((self.nloop * timesteps, self.N))
        Hx_vals = np.zeros((self.nloop * timesteps, self.N))
        errs = []
        rel_errs = []
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
                x_vals[itr + i * timesteps, :] = self.x
                Hx_vals[itr + i * timesteps, :] = self.Hx
                aux_targs[itr + i * timesteps, :] = np.dot(Jd, DRNN.Hx) + ufout[:, itr]

                # train connectivity matrix
                if np.random.rand() < 1/(self.train_every * self.dt):

                    Phx = np.dot(P, self.Hx)
     
                    # update correlation matrix
                    k = Phx / (1 + np.dot(np.transpose(self.Hx), Phx))
                    P = P - np.outer(Phx, k)
                    
                    # update error
                    err = np.dot(self.W_trained, self.Hx) - aux_targs[itr + i * timesteps, :] # error is vector
                    oerr = np.dot(self.W_out, self.Hx) - fout[itr] # error is scalar

                    # update connectivity
                    self.W_trained = self.W_trained - np.outer(err, k)

                    # update output weights
                    self.W_out = self.W_out - oerr * k
                    
                    # track training errors
                    errs.append(np.linalg.norm(err))
                    rel_errs.append(np.mean((np.dot(self.W_trained, self.Hx) - aux_targs[itr + i * timesteps, :]) / err))
                
        x_vals = np.transpose(x_vals)
        Hx_vals = np.transpose(Hx_vals)
        aux_targs = np.transpose(aux_targs)
        return x_vals, Hx_vals, errs, rel_errs, aux_targs, ufin, ufout