import numpy as np
import BBMM
import time
import unittest

np.random.seed(0)
lengthscale = 1.0 + np.random.random()
variance = 1.0 + np.random.random()


n = 3
d = 2
N = 20


def y(X):
    Y = np.sum(np.sin(X[:, 0:d]), axis=1)
    Y_grad = [np.sum(X[:, d * (i + 1):d * (i + 2)] * np.cos(X[:, d * (i + 1):d * (i + 2)]), axis=1) for i in range(n)]
    return np.concatenate([Y] + Y_grad)[:, None]


np.random.seed(0)
X = np.random.random((N, (n + 1) * d))
Y = y(X)
refs = {
    'RBF': np.array([2.21345653e+01, 9.05796792e+00, 7.13224985e-03]),
    'Matern32': np.array([1.16484618e+01, 1.15491463e+01, 4.35634565e-03]),
    'Matern52': np.array([8.94029960e+00, 8.07506130e+00, 6.98069283e-03]),
}
refs_factor = {
    'RBF': np.array([[2.99129653e+01, 9.34713942e+00, 8.81087658e-01, 6.87254269e-03]]),
    'Matern32': np.array([1.38463144e+01, 1.13619224e+01, 8.98054412e-01, 4.11836198e-03]),
    'Matern52': np.array([5.10313642e+01, 1.63221237e+01, 8.82278979e-01, 6.85344774e-03]),
}


class Test(unittest.TestCase):
    def _run_nofactor(self, kernelname):
        stationary_kernel = getattr(BBMM.kern, kernelname)()
        stationary_kernel.set_lengthscale(lengthscale)
        stationary_kernel.set_variance(variance)
        ka = BBMM.kern.FullDerivative(stationary_kernel, n, d, optfactor=False)
        gp = BBMM.GP(X, Y, ka, 1e-4)
        gp.optimize()
        err = (gp.params - refs[kernelname])/refs[kernelname]
        self.assertTrue(np.max(np.abs(err)) < 1e-6)

    def _run_factor(self, kernelname):
        print(kernelname)
        stationary_kernel = getattr(BBMM.kern, kernelname)()
        stationary_kernel.set_lengthscale(lengthscale)
        stationary_kernel.set_variance(variance)
        ka = BBMM.kern.FullDerivative(stationary_kernel, n, d, optfactor=True)
        gp = BBMM.GP(X, Y, ka, 1e-4)
        gp.optimize()
        err = (gp.params - refs_factor[kernelname])/refs_factor[kernelname]
        self.assertTrue(np.max(np.abs(err)) < 1e-6)

    def test_RBF(self):
        self._run_nofactor('RBF')

    def test_Matern32(self):
        self._run_nofactor('Matern32')

    def test_Matern52(self):
        self._run_nofactor('Matern52')

    def test_RBF_factor(self):
        self._run_factor('RBF')

    def test_Matern32_factor(self):
        self._run_factor('Matern32')

    def test_Matern52_factor(self):
        self._run_factor('Matern52')


if __name__ == '__main__':
    unittest.main()
