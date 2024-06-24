import numpy as np
import scipy as sp

def run_rate(rate_params, J, fin, fout): 
    # unpack parameters
    N, tau_x, gain, T, dt = rate_params

    # initialize variables
    x = np.zeros(N)

    # track variables
    x_vals = []

    # differential equation of dx/dt
    def dx(x):
        return 1/tau_x * (-x + gain * np.dot(J, np.tanh(x)) + fin + fout)
    
    t = 0
    while t < T: 
        # RK4 for each timestep
        x1 = dt * dx(x)
        x2 = dt * dx(x + x1/2)
        x3 = dt * dx(x + x2/2)
        x4 = dt * dx(x + x3)
        x_next = x + (x1 + 2*x2 + 2*x3 + x4) / 6

        x = x_next
        t += dt
        x_vals.append(x)
    
    return np.transpose(x_vals)