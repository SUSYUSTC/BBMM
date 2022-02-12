import numpy as np
import BBMM
import unittest

np.random.seed(0)
lengthscale = 1.0 + np.random.random()
variance = 1.0 + np.random.random()


n = 3
d = 2
N = 100


def y(X):
    Y = np.sum(np.sin(X[:, 0:d]), axis=1)
    Y_grad = [np.sum(X[:, d * (i + 1):d * (i + 2)] * np.cos(X[:, d * (i + 1):d * (i + 2)]), axis=1) for i in range(n)]
    return np.concatenate([Y] + Y_grad)[:, None]


np.random.seed(0)
X = np.random.random((N, (n + 1) * d))
Y = y(X)

ref = np.array([4.07525802e+01, 1.25747855e+01, 8.88520198e-03])
ref_factor = np.array([6.06428980e+01, 1.32487578e+01, 8.63056515e-01, 8.36132526e-03])
ref_order = np.array([5.01755444e+00, 2.73303199e+00, 1.00000000e-10, 3.33174327e-02])
ref_factor_order = np.array([5.02491805e+00, 2.73296982e+00, 8.65480936e-01, 1.00000000e-10, 1.88639965e-02])


class Test(unittest.TestCase):
    def _run(self, optfactor, split_type, ref):
        stationary_kernel = BBMM.kern.RBF()
        stationary_kernel.set_lengthscale(lengthscale)
        stationary_kernel.set_variance(variance)
        kernel = BBMM.kern.FullDerivative(stationary_kernel, n, d, optfactor=optfactor, likelihood_split_type = split_type)
        noise = [1e-4] * len(kernel.splits)
        gp = BBMM.GP(X, Y, kernel, noise)
        gp.optimize(messages=False)
        err = (gp.params - ref)/ref
        print(ref)
        print(gp.params)
        self.assertTrue(np.max(np.abs(err)) < 1e-5)

    def test(self):
        self._run(False, 'same', ref)

    def test_factor(self):
        self._run(True, 'same', ref_factor)

    def test_order(self):
        self._run(False, 'order', ref_order)

    def test_factor_order(self):
        self._run(True, 'order', ref_factor_order)


if __name__ == '__main__':
    unittest.main()
