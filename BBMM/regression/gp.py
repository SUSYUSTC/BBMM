import typing as tp
from collections.abc import Iterable
import numpy as np
from ..kern import Kernel, get_kern_obj, param_transformation
import time
import sys
from .noise import Noise
from .. import utils


class GP(object):
    def __init__(self, X: np.ndarray, Y: np.ndarray, kernel: Kernel, noise: tp.Union[utils.general_float, tp.Iterable[utils.general_float]], GPU: bool = False, file=None):
        self.Nin = len(X)
        self.kernel = kernel
        self.Nout = self.Nin * self.kernel.nout
        assert len(Y) == self.Nout
        self.noise = Noise(noise, self.kernel.n_likelihood_splits)
        self.likelihood_splits = self.kernel.split_likelihood(self.Nin)
        self.nks = len(self.kernel.ps)
        self.nns = len(self.noise.values)
        self.transformations_group = param_transformation.Group(kernel.transformations + [param_transformation.log] * self.nns)
        self.GPU = GPU
        if GPU:
            import cupy as cp
            self.xp = cp
            import cupyx
            import cupyx.scipy.linalg
            cupyx.seterr(linalg='raise')
            self.xp_solve_triangular = cupyx.scipy.linalg.solve_triangular
        else:
            import scipy
            self.xp = np
            self.xp_solve_triangular = scipy.linalg.solve_triangular
        self.X = self.xp.array(X).copy()
        self.Y = self.xp.array(Y).copy()
        if file is None:
            self.file = sys.__stdout__
        else:
            self.file = file

    def fit(self, grad: bool = False) -> None:
        self.grad = grad
        K_noise = self.kernel.K(self.X, cache=self.kernel.default_cache)
        diag_reg = self.noise.get_diag_reg(self.likelihood_splits)
        K_noise[self.xp.arange(self.Nout), self.xp.arange(self.Nout)] += self.xp.array(diag_reg)
        L = self.xp.linalg.cholesky(K_noise)
        del K_noise
        w_int = self.xp_solve_triangular(L, self.Y, lower=True, trans=0)
        self.w = self.xp_solve_triangular(L, w_int, lower=True, trans=1)
        if grad:
            logdet = self.xp.sum(self.xp.log(self.xp.diag(L))) * 2
            Linv = self.xp_solve_triangular(L, self.xp.eye(self.Nout), lower=True, trans=0)
            del L
            K_noise_inv = Linv.T.dot(Linv)
            del Linv
            self.ll = - (self.Y.T.dot(self.w)[0, 0] + logdet) / 2 - self.xp.log(self.xp.pi * 2) * self.Nout / 2
            dL_dK = (self.w.dot(self.w.T) - K_noise_inv) / 2
            del K_noise_inv
            dL_dps = [self.xp.sum(dL_dK * dK_dp(self.X, cache=self.kernel.default_cache)) for dK_dp in self.kernel.dK_dps]
            dL_dns = [self.xp.trace(dL_dK * self.xp.diag(dK_dn_diag)) for dK_dn_diag in self.noise.get_diag_reg_gradient(self.likelihood_splits)]
            self.gradient = self.xp.array(dL_dps + dL_dns)
            if self.GPU:
                self.ll = self.ll.get()
                self.gradient = self.gradient.get()

    def save(self, path: str) -> None:
        if self.GPU:
            data = {
                'kernel': self.kernel.to_dict(),
                'X': self.xp.asnumpy(self.X),
                'Y': self.xp.asnumpy(self.Y),
                'w': self.xp.asnumpy(self.w),
                'noise': self.noise.values,
                'grad': self.grad,
            }
        else:
            data = {
                'kernel': self.kernel.to_dict(),
                'X': self.X,
                'Y': self.Y,
                'w': self.w,
                'noise': self.noise.values,
                'grad': self.grad,
            }
        if self.grad:
            data['ll'] = self.ll
            data['gradient'] = self.gradient
        np.savez(path, **data)

    @classmethod
    def from_dict(self, data: tp.Dict[str, tp.Any], GPU: bool) -> 'GP':
        kernel_dict = data['kernel'][()]
        kernel = get_kern_obj(kernel_dict)
        result = self(data['X'], data['Y'], kernel, noise=data['noise'][()], GPU=GPU)
        if GPU:
            result.w = result.xp.asarray(data['w'])
        else:
            result.w = data['w']
        return result

    @classmethod
    def load(self, path: str, GPU: bool) -> 'GP':
        data = dict(np.load(path, allow_pickle=True))
        return self.from_dict(data, GPU)

    def predict(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        self.kernel.clear_cache()
        if self.GPU:
            result = self.xp.asnumpy(self.kernel.K(self.xp.asarray(X), self.X).dot(self.w))
            if training:
                result += self.xp.asnumpy(self.w) * self.noise.get_diag_reg(self.likelihood_splits)[:, None]
            return result
        else:
            result = self.kernel.K(X, self.X).dot(self.w)
            if training:
                result += self.w * self.noise.get_diag_reg(self.likelihood_splits)[:, None]
            return result

    def update(self, ps: tp.List[utils.general_float], noises: tp.List[float]) -> None:
        self.kernel.set_all_ps(ps)
        self.noise.values = noises
        self.kernel.clear_cache()
        self.params = np.array(ps + noises)

    def objective(self, transform_ps_noise: np.ndarray) -> tp.Tuple[float, np.ndarray]:
        ps_noise = self.transformations_group.inv(transform_ps_noise.tolist())
        if self.messages:
            print('x:' + ' %e' * len(ps_noise) % tuple(ps_noise), file=self.file, flush=True)
        d_transform_ps_noise = self.transformations_group.d(ps_noise)
        self.update(ps_noise[:self.nks], ps_noise[-self.nns:])
        self.fit(grad=True)
        self.transform_gradient = self.gradient / np.array(d_transform_ps_noise)
        result = (-self.ll, -self.transform_gradient)
        return result

    def get_numerical_gradient(self, transform_ps_noise: np.ndarray, delta: float) -> np.ndarray:
        n_grad = []
        for i in range(len(transform_ps_noise)):
            transform_ps_noise_p = transform_ps_noise.copy()
            transform_ps_noise_p[i] += delta
            obj_p, _ = self.objective(transform_ps_noise_p)
            transform_ps_noise_n = transform_ps_noise.copy()
            transform_ps_noise_n[i] -= delta
            obj_n, _ = self.objective(transform_ps_noise_n)
            n_grad.append((obj_p-obj_n)/(delta*2))
        return np.array(n_grad)

    def opt_callback(self, x):
        if self.messages:
            print('-ll', np.format_float_scientific(-self.ll, precision=6), 'gradient', np.linalg.norm(self.transform_gradient), file=self.file, flush=True)
            print(file=self.file, flush=True)
        self.saved_ps = list(map(lambda p: p.value, self.kernel.ps))
        self.saved_noises = self.noise.values
        update = False
        if not hasattr(self, 'current_best_ll'):
            update = True
        else:
            if self.ll > self.current_best_ll:
                update = True
        if update:
            self.current_best_ll = self.ll
            self.current_best_ps = self.saved_ps
            self.current_best_noises = self.saved_noises

    def optimize(self, messages=False, tol=1e-6, noise_bound: tp.Union[utils.general_float, tp.List[utils.general_float]] = 1e-10) -> None:
        import scipy
        import scipy.optimize
        self.messages = messages
        callback: tp.Callable = self.opt_callback
        begin = time.time()
        noise_bound_list = utils.make_desired_size(noise_bound, self.kernel.n_likelihood_splits)
        import warnings
        n_try = 1
        while True:
            try:
                ps_noise = list(map(lambda p: p.value, self.kernel.ps)) + self.noise.values
                transform_ps_noise = self.transformations_group(ps_noise)
                bounds: tp.List[tp.Tuple[float, float]] = [(-np.inf, np.inf) for i in range(self.nks)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for b in noise_bound_list:
                        bounds.append((float(np.log(b)), np.inf))
                self.result = scipy.optimize.minimize(self.objective, transform_ps_noise, jac=True, method='L-BFGS-B', callback=callback, tol=tol, bounds=bounds, options={'maxls': 100})
                if self.result.success or (n_try >= 12):
                    break
                print("Optimization not successful, restarting", file=self.file, flush=True)
                print("current x", self.result.x, file=self.file, flush=True)
                print("current grad", self.result.jac, file=self.file, flush=True)
                print("current p", -self.result.hess_inv.dot(self.result.jac), file=self.file, flush=True)
                n_try += 1
            except np.linalg.LinAlgError:
                noise_bound_list = [item * 2 for item in noise_bound_list]
                print('Cholesky decomposition failed. Try to use a higher noise bound', noise_bound_list, file=self.file, flush=True)
                self.update(self.saved_ps, self.saved_noises)

        self.success = self.result.success
        if self.ll < self.current_best_ll:
            self.update(self.current_best_ps, self.current_best_noises)
            self.fit(grad=True)
            print('Optimization failed, taking the best history value, -ll =', -self.ll, file=self.file, flush=True)
        end = time.time()
        print('time', end - begin, file=self.file, flush=True)
