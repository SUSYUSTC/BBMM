import numpy as np
import BBMM
import os
import unittest


class Test(unittest.TestCase):
    def _run_GP(self, GPU):
        X = np.load("./X_test.npy")
        Y = np.load("./Y_test.npy")
        noise = 1e-5
        k = BBMM.kern.RBF()
        lengthscale = 1.0
        variance = 1.0
        k.set_lengthscale(lengthscale)
        k.set_variance(variance)
        gp = BBMM.GP(X, Y, k, noise, GPU=GPU)
        gp.fit()
        gp.save("model_GP")
        gp_load = BBMM.GP.load("model_GP.npz", GPU)
        err = np.max(np.abs(gp.predict(X, training=True) - Y))
        err_load = np.max(np.abs(gp_load.predict(X, training=True) - Y))
        self.assertTrue(err < 1e-8)
        self.assertTrue(err_load < 1e-8)
        os.remove("model_GP.npz")

    def _run_BBMM(self, GPU):
        X = np.load("./X_test.npy")
        Y = np.load("./Y_test.npy")
        noise = 1e-5
        k = BBMM.kern.RBF()
        lengthscale = 1.0
        variance = 1.0
        k.set_lengthscale(lengthscale)
        k.set_variance(variance)
        if GPU:
            nGPU = 1
        else:
            nGPU = 0
        bbmm = BBMM.BBMM(k, nGPU, verbose=False)
        bbmm.initialize(X, noise)
        bbmm.set_preconditioner(len(X)//5)
        bbmm.solve_iter(Y)
        bbmm.save("model_BBMM")
        bbmm_load = BBMM.BBMM.load("model_BBMM.npz", GPU)
        pred = bbmm.predict(X)
        pred_load = bbmm_load.predict(X)
        err = np.max(np.abs(pred - pred_load))
        self.assertTrue(err < 1e-8)
        os.remove("model_BBMM.npz")

    def test_CPU_GP(self):
        self._run_GP(False)

    def test_GPU_GP(self):
        self._run_GP(True)

    def test_CPU_BBMM(self):
        self._run_BBMM(False)

    def test_GPU_BBMM(self):
        self._run_BBMM(True)


if __name__ == '__main__':
    unittest.main()
