import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

class QIF_training(spike_training):
    
    def create_default_params():
        neuron_params = {
            'network_size': 200, # units in network
            'tau': 1, # ms, neuron decay constant
            'tau_s': 20, # ms, synaptic filtering constant
            'lam': 1 # learning rate factor
        }
        time_params = {
            'total_time': 1000, # ms, total runtime 
            'stim_on': 0, # ms, start of stimulus on
            'stim_off': 50, # ms, stim off time
            'dt': 0.1 # ms, timestep for differential equations
        }
        train_params = {
            'training_loops': 10, # number of training loops
            'train_every': 2 # ms, timestep of updating connectivity matrix
        }
        return neuron_params, time_params, train_params


    # ODE of theta neuron model
    def dtheta(theta, n_drive):
        return 1/tau * (1 - np.cos(theta) + np.multiply(n_drive, (1 + np.cos(theta))))

    # ODE of filtered spike train
    def dr(r): 
        return -1/tau_s * r 
        # does not include addition of new spikes
        
    def train_QIF(self, neuron_params, time_params, train_params, W, stim, targets):
            
        # unpack parameters
        N, tau, tau_s, lam = neuron_params
        T, stim_on, stim_off, dt = time_params
        nloop, train_every = train_params
        
        # tau: neuron decay constant
        # tau_s: synaptic decay constant
        # lam: learning rate
        # T: total training time
        # dt: time step
        # nloop: number of training trials

        # initialize variables
        theta = np.zeros(N) # phase of neurons
        u = np.zeros(N) # synaptic drive
        r = np.zeros(N) # filtered synaptic drive
        spk_t = np.zeros(N) # tracks spikes
        
        # initialize correlation matrices for each row
        Pidx = np.zeros((N, N))
        Ps = []
        for row in range(N):
            Pidx[row] = (W[row] != 0)
            tot = int(np.sum(Pidx[row]))
            Ps.append(1/lam * np.identity(tot))
        Pidx = Pidx.astype(bool)

        # optional variables to track states
        spks = []
        sdrive = []
        thetas = []
        spk_rast = []
        
        # training loop
        for i in range(nloop):
            print(i)
            t = 0.0
            itr = 0
            while t < T:

                # RK4 for theta neuron model (one step)
                k1 = dt * dtheta(theta, u + stim[:, int(t/dt)]);
                k2 = dt * dtheta(theta + k1/2, u + stim[:, int(t/dt)]);
                k3 = dt * dtheta(theta + k2/2, u + stim[:, int(t/dt)]);
                k4 = dt * dtheta(theta + k3, u + stim[:, int(t/dt)]);
                theta_next = theta + (k1 + 2*k2 + 2*k3 + k4)/6;

                # RK4 for filtered spike train (one step)
                r1 = dt * dr(r) 
                r2 = dt * dr(r + r1/2)
                r3 = dt * dr(r + r2/2)
                r4 = dt * dr(r + r3)
                r_next = r + (r1 + 2*r2 + 2*r3 + r4)/6

                # spike detection
                idx1 = theta_next - theta > 0
                idx2 = theta_next - theta > np.mod(np.pi - theta, 2*np.pi) # surely there's a better way to do this...
                idx = np.multiply(idx1, idx2)
                r[idx] += 1/tau_s # update spikes in r
                spk_t[idx] += 1

                # update variables
                theta = np.mod(theta + (k1 + 2*k2 + 2*k3 + k4)/6, 2*np.pi)
                r = r + (r1 + 2*r2 + 2*r3 + r4)/6
                u = np.dot(W, r)
                t = t + dt
                
                # track variables (optional)
                spks.append(r)
                sdrive.append(u)  
                thetas.append(theta)
                spk_rast.append(spk_t)
                spk_t = np.zeros(N)

                # train W matrix
                if t > stim_off and t < int(T) and np.mod(int(t/dt), int(train_every/dt)) == 0: # only train after initial stimulus
                    for row in range(N): # update each row of W by RLS

                        # update correlation matrix
                        numer = np.outer(np.dot(Ps[row], r[Pidx[row]]), np.dot(Ps[row], r[Pidx[row]]))
                        denom = 1 + np.dot(r[Pidx[row]], np.dot(Ps[row], r[Pidx[row]]))
                        Ps[row] = Ps[row] - numer / denom

                        # update error term
                        err = targets[row][int(t/dt)] - np.dot(W[row][Pidx[row]], r[Pidx[row]]) # error is scalar

                        # update W
                        W[row][Pidx[row]] = W[row][Pidx[row]] + err * np.dot(Ps[row], r[Pidx[row]])
            
        spks = np.transpose(spks)
        sdrive = np.transpose(sdrive)
        thetas = np.transpose(thetas)
        spk_rast = np.transpose(spk_rast)
        return W, spks, sdrive, thetas, spk_rast

    def run_QIF(self, neuron_params, time_params, W, stim, run_time):

        # unpack parameters
        N, tau, tau_s, lam = neuron_params
        T, stim_on, stim_off, dt = time_params
        
        # tau: neuron decay constant
        # tau_s: synaptic decay constant
        # lam: learning rate
        # T: total training time
        # dt: time step
        # nloop: number of training trials

        # initialize variables
        theta = np.zeros(N) # phase of neurons
        u = np.zeros(N) # synaptic drive
        r = np.zeros(N) # filtered synaptic drive
        spk_t = np.zeros(N) # discrete spikes
        
        # ODE of theta neuron model
        def dtheta(theta, n_drive):
            return 1/tau * (1 - np.cos(theta) + np.multiply(n_drive, (1 + np.cos(theta))))

        # ODE of filtered spike train
        def dr(r): 
            return -1/tau_s * r 
            # does not include addition of new spikes

        # optional variables to track states
        spks = []
        sdrive = []
        thetas = []
        spk_rast = []

        # training loop
        t = 0.0
        itr = 0
        while t < run_time:

            # RK4 for theta neuron model (one step)
            if t < stim_off:
                ext = stim[:, int(t/dt)]
            if t > stim_off:
                ext = np.zeros(N)
            
            k1 = dt * dtheta(theta, u + ext);
            k2 = dt * dtheta(theta + k1/2, u + ext);
            k3 = dt * dtheta(theta + k2/2, u + ext);
            k4 = dt * dtheta(theta + k3, u + ext);
            theta_next = theta + (k1 + 2*k2 + 2*k3 + k4)/6;

            # RK4 for filtered spike train (one step)
            r1 = dt * dr(r) 
            r2 = dt * dr(r + r1/2)
            r3 = dt * dr(r + r2/2)
            r4 = dt * dr(r + r3)
            r_next = r + (r1 + 2*r2 + 2*r3 + r4)/6

            # spike detection
            idx1 = theta_next - theta > 0
            idx2 = theta_next - theta > np.mod(np.pi - theta, 2*np.pi) # surely there's a better way to do this...
            idx = np.multiply(idx1, idx2)
            r[idx] += 1/tau_s # update spikes in r
            spk_t[idx] += 1
            
            # update variables
            theta = np.mod(theta + (k1 + 2*k2 + 2*k3 + k4)/6, 2*np.pi)
            r = r + (r1 + 2*r2 + 2*r3 + r4)/6
            u = np.dot(W, r)
            t = t + dt
            
            # track variables (optional)
            spks.append(r)
            sdrive.append(u)  
            thetas.append(theta)
            spk_rast.append(spk_t)
            spk_t = np.zeros(N)

        spks = np.transpose(spks)
        sdrive = np.transpose(sdrive)
        thetas = np.transpose(thetas)
        spk_rast = np.transpose(spk_rast)
        return spks, sdrive, thetas, spk_rast