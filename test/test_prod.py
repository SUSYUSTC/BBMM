import BBMM
import numpy as np
import GPy
import unittest

rbf = BBMM.kern.RBF()
matern32 = BBMM.kern.Matern32()
np.random.seed(0)
N = 10
d = 5
X = np.random.random((N, d * 2))
Y = np.sum(np.sin(X), axis=1)[:, None]
noise = 1e-4


class Test(unittest.TestCase):
    def setUp(self):
        prod = BBMM.kern.ProductKernel([rbf, matern32], dims=[slice(0, d), slice(d, 2 * d)])
        self.gp = BBMM.GP(X, Y, prod, noise)
        self.gp.optimize()
        kern = GPy.kern.Prod([GPy.kern.RBF(d, active_dims=np.arange(0, d)), GPy.kern.Matern32(d, active_dims=np.arange(d, 2 * d))])
        self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise)
        self.model.optimize()

    def test(self):
        err = np.max(np.abs(self.gp.params - self.model.param_array))
        print(err)
        self.assertTrue(err < 5e-3)


if __name__ == '__main__':
    unittest.main()
