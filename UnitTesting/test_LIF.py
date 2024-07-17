import numpy as np
import cupy as cp
import scipy as sp
import matplotlib.pyplot as plt
import unittest

from cProfile import Profile

from LIFTraining import LIFTraining
from LIFTraining import create_default_params_LIF

class TestLIFModel(unittest.TestCase):

    def test_runCPU(self):
        p = create_default_params_LIF()
        testnet = LIFTraining(p)
        stim = testnet.gen_rand_stim(0, 0)
        self.assertEqual(np.shape(stim)[1], int(testnet.T / testnet.dt))
        x, Hx, voltage = testnet.run(stim)
        
        r = testnet.run_time - 1

        for i in range(testnet.N):
            self.assertTrue(x[i, r] < 0)