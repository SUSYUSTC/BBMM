import BBMM
import numpy as np
import sys
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
lr = 0.5
opt = BBMM.Optimizer(lengthscale, variance, noise, clamp_noise=1e-5, init_lr=lr)
while True:
    kernel = BBMM.RBF()
    bbmm = BBMM.BBMM(kernel, nGPU=1, file=None, verbose=False)
    bbmm.initialize(X, opt.lengthscale, opt.variance, opt.noise, batch=batch)
    bbmm.set_preconditioner(N_init, nGPU=0)
    # must use the same random seed through the optimization!
    bbmm.solve_iter(Y, thres=thres, block_size=bs, compute_gradient=True, random_seed=0, compute_loglikelihood=False, lanczos_n_iter=20, debug=False, max_iter=1000)
    assert bbmm.converged
    opt.step(-bbmm.gradients.lengthscale, -bbmm.gradients.variance, -bbmm.gradients.noise)
    print(opt.history_grads[-1], [opt.lengthscale, opt.variance, opt.noise])
    if len(opt.history_parameters) >= 5:
        # decrease learing rate if the recent fluctuation is small
        recent = np.array(opt.history_parameters[-5:])
        fluc = np.max(recent, axis=0) - np.min(recent, axis=0)
        if np.all(fluc < lr * 0.5):
            opt.set_lr(lr * 0.5)
            lr *= 0.5
            print("set learning rate to", lr)
    if lr < 1e-2:
        break

print("optimized paramters:")
print("lengthscale:", opt.lengthscale)
print("variance:", opt.variance)
print("noise:", opt.noise)
thres = 1e-8
kernel = BBMM.RBF()
bbmm = BBMM.BBMM(kernel, nGPU=1, file=None, verbose=False)
bbmm.initialize(X, opt.lengthscale, opt.variance, opt.noise, batch=batch)
bbmm.set_preconditioner(N_init, nGPU=0)
bbmm.verbose = True
bbmm.set_preconditioner(N_init, nGPU=0)
woodbuery_vec_iter = bbmm.solve_iter(Y, thres=thres, block_size=bs, compute_gradient=True, random_seed=0, compute_loglikelihood=False, lanczos_n_iter=20, debug=False, max_iter=1000)
assert bbmm.converged
pred = bbmm.predict(X, woodbuery_vec_iter, training=True)
print("MAX Error:", np.max(np.abs(pred - Y)))
