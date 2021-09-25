import numpy as np
import time
import BBMM
import GPy
import unittest


class Test(unittest.TestCase):
    def _run(self, GPU):
        X = np.load("./X_test.npy")
        Y = np.load("./Y_test.npy")
        noise = 1e-5
        k = BBMM.kern.RBF()
        lengthscale = 1.0
        variance = 1.0
        k.set_lengthscale(lengthscale)
        k.set_variance(variance)
        begin = time.time()
        gp = BBMM.GP(X, Y, k, noise, GPU=GPU)
        gp.optimize(messages=False)
        print("BBMM GP Time", time.time() - begin)

        GPy_kern = GPy.kern.RBF(input_dim=X.shape[1], lengthscale=lengthscale, variance=variance)
        begin = time.time()
        model = GPy.models.GPRegression(X, Y, kernel=GPy_kern, noise_var=noise)
        model.optimize(messages=False)
        print("GPy Time", time.time() - begin)

        err = np.max(np.abs((model.param_array - gp.params) / model.param_array))
        self.assertTrue(err < 0.01)

    def test_CPU(self):
        self._run(False)

    def test_GPU(self):
        self._run(True)


if __name__ == '__main__':
    unittest.main()
