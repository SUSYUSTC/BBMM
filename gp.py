import numpy as np
import cupy as cp
import time


class GP(object):
    def __init__(self, X, Y, kernel, noise=1e-4, GPU=False):
        self.kernel = kernel
        self.noise = noise
        self.GPU = GPU
        if GPU:
            self.xp = cp
            import cupyx
            import cupyx.scipy.linalg
            self.xp_solve_triangular = cupyx.scipy.linalg.solve_triangular
        else:
            import scipy
            self.xp = np
            self.xp_solve_triangular = scipy.linalg.solve_triangular
        self.X = self.xp.array(X).copy()
        self.Y = self.xp.array(Y).copy()
        self.N = len(Y)

    def fit(self, grad=False):
        K = self.kernel.K(self.X, cache=self.kernel.default_cache)
        K_noise = K + self.xp.eye(self.N) * self.noise
        L = self.xp.linalg.cholesky(K_noise)
        w_int = self.xp_solve_triangular(L, self.Y, lower=True, trans=0)
        self.w = self.xp_solve_triangular(L, w_int, lower=True, trans=1)
        if grad:
            Linv = self.xp_solve_triangular(L, self.xp.eye(self.N), lower=True, trans=0)
            K_noise_inv = Linv.T.dot(Linv)
            logdet = self.xp.linalg.slogdet(K_noise)[1]
            self.ll = - (self.Y.T.dot(self.w)[0, 0] + logdet) / 2 - self.xp.log(self.xp.pi * 2) * self.N / 2
            dL_dK = (self.w.dot(self.w.T) - K_noise_inv) / 2
            dL_dl = self.xp.sum(dL_dK * self.kernel.dK_dl(self.X, cache=self.kernel.default_cache))
            dL_dv = self.xp.sum(dL_dK * self.kernel.dK_dv(self.X, cache=self.kernel.default_cache))
            dL_dn = self.xp.trace(dL_dK)
            self.gradient = self.xp.array([dL_dv, dL_dl, dL_dn])
            if self.GPU:
                self.ll = self.ll.get()
                self.gradient = self.gradient.get()

    def predict(self, X):
        return self.kernel.K(X, self.X, cache={}).dot(self.w)

    def update(self, lengthscale, variance, noise):
        self.kernel.set_lengthscale(lengthscale)
        self.kernel.set_variance(variance)
        self.noise = noise
        self.kernel.clear_cache()
        self.params = np.array([variance, lengthscale, noise])

    def objective(self, ps):
        variance = np.exp(ps[0])
        lengthscale = np.exp(ps[1])
        noise = np.exp(ps[2])
        self.update(lengthscale, variance, noise)
        self.fit(grad=True)
        result = (-self.ll, -self.gradient * np.exp(ps))
        return result

    def opt_callback(self, x):
        print('ll', np.format_float_scientific(-self.ll, precision=6), 'x', x, -self.gradient * np.exp(x))

    def optimize(self, messages=False, tol=1e-6):
        import scipy
        if messages:
            callback = self.opt_callback
        else:
            callback = None
        begin = time.time()
        self.result = scipy.optimize.minimize(self.objective, np.log([self.kernel.variance, self.kernel.lengthscale, self.noise]), jac=True, method='L-BFGS-B', callback=callback, tol=tol)
        end = time.time()
        print('time', end - begin)
