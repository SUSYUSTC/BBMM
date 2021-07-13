import BBMM
import numpy as np
import sys
import unittest

dtype = np.float64
X = np.load("./X_test.npy").astype(dtype)
Y = np.load("./Y_test.npy").astype(dtype)
N = len(Y)
lengthscale = 1.0
variance = 10.0
noise = 1e-2
batch = min(4096, N)
thres = 1e-6
# larger block size gives more accurate gradient
bs = 50
N_init = 500


class Test(unittest.TestCase):
    def test(self):
        lr = 0.5
        opt = BBMM.Adam(lengthscale, variance, noise, clamp_noise=1e-5, init_lr=lr)
        while True:
            kernel = BBMM.RBF()
            bbmm = BBMM.BBMM(kernel, nGPU=1, file=None, verbose=False)
            bbmm.initialize(X, opt.lengthscale, opt.variance, opt.noise, batch=batch)
            bbmm.set_preconditioner(N_init, nGPU=0)
            # must use the same random seed through the optimization!
            bbmm.solve_iter(Y, thres=thres, block_size=bs, compute_gradient=True, random_seed=0, compute_loglikelihood=False, lanczos_n_iter=20, debug=False, max_iter=1000)
            self.assertTrue(bbmm.converged)
            opt.step(-bbmm.gradients.lengthscale, -bbmm.gradients.variance, -bbmm.gradients.noise)
            #print(opt.history_grads[-1], [opt.lengthscale, opt.variance, opt.noise], end='\r')
            print(np.linalg.norm(opt.history_grads[-1]), end='\r')
            if len(opt.history_parameters) >= 5:
                # decrease learing rate if the recent fluctuation is small
                recent = np.array(opt.history_parameters[-5:])
                fluc = np.max(recent, axis=0) - np.min(recent, axis=0)
                if np.all(fluc < lr * 0.5):
                    opt.set_lr(lr * 0.5)
                    lr *= 0.5
                    #print("set learning rate to", lr)
            if lr < 1e-2:
                break

        self.assertTrue(np.abs(opt.lengthscale - 3.15) < 0.1)
        self.assertTrue(np.abs(opt.variance - 1300) < 50)
        self.assertTrue(np.abs(opt.noise - 0.013) < 0.001)
        kernel = BBMM.RBF()
        bbmm = BBMM.BBMM(kernel, nGPU=1, file=None, verbose=False)
        bbmm.initialize(X, opt.lengthscale, opt.variance, opt.noise, batch=batch)
        bbmm.set_preconditioner(N_init, nGPU=0)
        bbmm.verbose = False
        bbmm.set_preconditioner(N_init, nGPU=0)
        woodbuery_vec_iter = bbmm.solve_iter(Y, thres=thres, block_size=bs, compute_gradient=True, random_seed=0, compute_loglikelihood=False, lanczos_n_iter=20, debug=False, max_iter=1000)
        self.assertTrue(bbmm.converged)
        pred = bbmm.predict(X, woodbuery_vec_iter, training=True)
        err = np.max(np.abs(pred - Y))
        self.assertTrue(err < 1e-6)


if __name__ == '__main__':
    unittest.main()