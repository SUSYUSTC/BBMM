import BBMM
import numpy as np
dtype = np.float64
X = np.load("./X_test.npy").astype(dtype)
Y = np.load("./Y_test.npy").astype(dtype)
N = len(Y)
lengthscale = 1.0
variance = 5.0
noise = 1e-2
batch = min(4096, N)
thres = 1e-8
# larger block size gives more accurate gradient
bs = 1
N_init = 500
lr = 0.5
kernel = BBMM.RBF()
bbmm = BBMM.BBMM(kernel, nGPU=1, file=None, verbose=False)
#bbmm.initialize(X, lengthscale, variance, noise, batch=batch, init_lr=lr, clamp_noise=1e-5)
bbmm.initialize(X, lengthscale, variance, noise, batch=batch)
bbmm.verbose = True
bbmm.set_preconditioner(N_init, nGPU=0)
woodbuery_vec_iter = bbmm.solve_iter(Y, thres=thres, block_size=bs, compute_gradient=True, random_seed=0, compute_loglikelihood=False, lanczos_n_iter=20, debug=False, max_iter=1000)
assert bbmm.converged
pred = bbmm.predict(X, woodbuery_vec_iter, training=True)
print("MAX Error:", np.max(np.abs(pred - Y)))
