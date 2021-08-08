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


class Test(unittest.TestCase):
    def _run(self, kernelname):
        stationary_kernel = getattr(BBMM.kern, kernelname)()
        stationary_kernel.set_lengthscale(lengthscale)
        stationary_kernel.set_variance(variance)
        ka = BBMM.kern.FullDerivative(stationary_kernel, n, d)
        gp = BBMM.GP(X, Y, ka, 1e-4)
        gp.optimize()
        err = gp.params - refs[kernelname]
        self.assertTrue(np.max(np.abs(err)/refs[kernelname]) < 1e-8)

    def test_RBF(self):
        self._run('RBF')

    def test_Matern32(self):
        self._run('Matern32')

    def test_Matern52(self):
        self._run('Matern52')


if __name__ == '__main__':
    unittest.main()
