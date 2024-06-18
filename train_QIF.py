def train_QIF(neuron_params, time_params, W, stim, targets):

    # unpack parameters
    tau, tau_s, lam = neuron_params
    T, stim_on, stim_off, dt = time_params

    # tau: neuron decay constant
    # tau_s: synaptic decay constant
    # lam: learning rate
    # T: total training time
    # dt: time step

    # initialize variables
    theta = np.zeros(N) # phase of neurons
    u = np.zeros(N) # synaptic drive
    r = np.zeros(N) # filtered synaptic drive

    # initialize correlation matrices for each row
    Pidx = np.zeros((N, N))
    Ps = []
    for row in range(N):
        Pidx[row] = (W[row] != 0)
        tot = int(np.sum(Pidx[row]))
        Ps.append(1/lam * np.identity(tot))
    Pidx = Pidx.astype(bool)

    # ODE of theta neuron model
    def dtheta(theta, n_drive):
        return 1/tau * (1 - np.cos(theta) + np.multiply(n_drive, (1 + np.cos(theta))))

    # ODE of filtered spike train
    def dr(r): 
        return -1/tau_s * r 
        # does not include addition of new spikes

    # training loop
    t = 0.0
    while t < T - dt:

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
        idx2 = theta_next - theta > np.mod(np.pi - np.mod(theta, 2*np.pi), 2*np.pi) # surely there's a better way to do this...
        idx = np.multiply(idx1, idx2)
        r[idx] += 1/tau_s # update spikes in r
        
        # update variables
        theta = np.mod(theta + (k1 + 2*k2 + 2*k3 + k4)/6, 2*np.pi)
        r = r + (r1 + 2*r2 + 2*r3 + r4)/6
        u = np.dot(W, r)
        t = t + dt
    
        # train W matrix
        if t > stim_off: # only train after initial stimulus
            for row in range(N): # update each row of W by RLS
                
                # update correlation matrix
                k = Ps[row] * r[Pidx[row]] # just a temp variable 
                numer = k*np.transpose(r[Pidx[row]])*Ps[row]
                denom = 1 + np.transpose(r[Pidx[row]])*k
                Ps[row] = Ps[row] - numer / denom
                
                # update error term
                err = targets[row][int(t/dt)] - np.dot(W[row][Pidx[row]], r[Pidx[row]]) # error is scalar
                
                # update W
                W[row][Pidx[row]] = W[row][Pidx[row]] + err * np.dot(Ps[row], r[Pidx[row]])
                
    return W