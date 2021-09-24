import numpy as np
from . import kern
import time
import sys
from . import param_transformation


class GP(object):
    def __init__(self, X, Y, kernel, noise, GPU=False, file=None):
        self.kernel = kernel
        self.transformations_group = param_transformation.Group(kernel.transformations + [param_transformation.log])
        self.noise = noise
        self.GPU = GPU
        if GPU:
            import cupy as cp
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
        if file is None:
            self.file = sys.__stdout__
        else:
            self.file = file

    def fit(self, grad=False):
        self.grad = grad
        K_noise = self.kernel.K(self.X, cache=self.kernel.default_cache) + self.xp.eye(self.N) * self.noise
        L = self.xp.linalg.cholesky(K_noise)
        w_int = self.xp_solve_triangular(L, self.Y, lower=True, trans=0)
        self.w = self.xp_solve_triangular(L, w_int, lower=True, trans=1)
        if grad:
            #logdet = self.xp.linalg.slogdet(K_noise)[1]
            logdet = np.sum(np.log(np.diag(L)))*2
            del K_noise
            Linv = self.xp_solve_triangular(L, self.xp.eye(self.N), lower=True, trans=0)
            del L
            K_noise_inv = Linv.T.dot(Linv)
            del Linv
            self.ll = - (self.Y.T.dot(self.w)[0, 0] + logdet) / 2 - self.xp.log(self.xp.pi * 2) * self.N / 2
            dL_dK = (self.w.dot(self.w.T) - K_noise_inv) / 2
            del K_noise_inv
            dL_dps = [self.xp.sum(dL_dK * dK_dp(self.X, cache=self.kernel.default_cache)) for dK_dp in self.kernel.dK_dps]
            dL_dn = self.xp.trace(dL_dK)
            self.gradient = self.xp.array(dL_dps + [dL_dn])
            if self.GPU:
                self.ll = self.ll.get()
                self.gradient = self.gradient.get()

    def save(self, path):
        if self.GPU:
            data = {
                'kernel': self.kernel.to_dict(),
                'X': self.xp.asnumpy(self.X),
                'Y': self.xp.asnumpy(self.Y),
                'w': self.xp.asnumpy(self.w),
                'noise': self.noise,
                'grad': self.grad,
            }
        else:
            data = {
                'kernel': self.kernel.to_dict(),
                'X': self.X,
                'Y': self.Y,
                'w': self.w,
                'noise': self.noise,
                'grad': self.grad,
            }
        if self.grad:
            data['ll'] = self.ll
            data['gradient'] = self.gradient
        np.savez(path, **data)

    @classmethod
    def from_dict(self, data, GPU):
        kernel_dict = data['kernel'][()]
        kernel = kern.get_kern_obj(kernel_dict)
        result = GP(data['X'], data['Y'], kernel, noise=data['noise'][()], GPU=GPU)
        if GPU:
            result.w = result.xp.asarray(data['w'])
        else:
            result.w = data['w']
        return result

    @classmethod
    def load(self, path, GPU):
        data = dict(np.load(path, allow_pickle=True))
        return self.from_dict(data, GPU)

    def predict(self, X, training=False):
        self.kernel.clear_cache()
        if self.GPU:
            result = self.xp.asnumpy(self.kernel.K(self.xp.asarray(X), self.X).dot(self.w))
            if training:
                result += self.xp.asnumpy(self.w) * self.noise
            return result
        else:
            result = self.kernel.K(X, self.X).dot(self.w)
            if training:
                result += self.w * self.noise
            return result

    def update(self, ps, noise):
        self.kernel.set_all_ps(ps)
        self.noise = noise
        self.kernel.clear_cache()
        self.params = np.array(ps + [noise])

    def objective(self, transform_ps_noise):
        ps_noise = self.transformations_group.inv(transform_ps_noise)
        d_transform_ps_noise = self.transformations_group.d(ps_noise)
        self.update(ps_noise[:-1], ps_noise[-1])
        self.fit(grad=True)
        self.transform_gradient = self.gradient / np.array(d_transform_ps_noise)
        result = (-self.ll, -self.transform_gradient)
        return result

    def opt_callback(self, x):
        print('ll', np.format_float_scientific(-self.ll, precision=6), 'gradient', np.linalg.norm(self.transform_gradient), file=self.file, flush=True)

    def optimize(self, messages=False, tol=1e-6):
        import scipy
        import scipy.optimize
        if messages:
            callback = self.opt_callback
        else:
            callback = None
        begin = time.time()
        transform_ps = [self.kernel.transformations[i](self.kernel.ps[i].value) for i in range(len(self.kernel.ps))]
        transform_noise = np.log(self.noise)
        self.result = scipy.optimize.minimize(self.objective, transform_ps + [transform_noise], jac=True, method='L-BFGS-B', callback=callback, tol=tol)
        end = time.time()
        print('time', end - begin, file=self.file, flush=True)
