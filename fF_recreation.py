from rate_training.py import *

def fF():
    npar, tpar, trpar, cpar, rpar = create_default_params()
    DRNN = rate_training(npar, tpar, trpar, cpar, rpar)
    TRNN = rate_training(npar, tpar, trpar, cpar, rpar)

    