# Gaussian Process (GP) and Revised BBMM methods for GP

## Installation ##
```python setup.py install```

## Dependences ##
* cupy ```pip install cupy ``` 

## How to use ##
Train a Full GP model and then save
```
import numpy as np
import BBMM
# create data and kernel
X = np.random.random((100, 10))
Y = np.sum(np.sin(X), axis=1)[:, None]
noise = 1e-4
k = BBMM.kern.RBF()
k.set_lengthscale(1.0)
k.set_variance(10.0)
# train
gp = BBMM.GP(X, Y, k, noise, GPU=False)
gp.optimize(messages=False)
# predict
Y_pred = bbmm.pred(X)
# save
gp.save("model.npz")
```

Train a BBMM model and then save
```
import numpy as np
import BBMM
# create data and kernel
X = np.random.random((100, 10))
Y = np.sum(np.sin(X), axis=1)[:, None]
noise = 1e-4
k = BBMM.kern.RBF()
k.set_lengthscale(1.0)
k.set_variance(10.0)
# train
bbmm = BBMM.BBMM(k, nGPU=1)
bbmm.initialize(X, noise)
bbmm.set_preconditioner(50, nGPU=0)
bbmm.solve_iter(Y)
# predict
Y_pred = bbmm.pred(X)
# save
bbmm.save("model.npz")
```

## Referencess ##
1. Wang, Ke Alexander, Geoff Pleiss, Jacob R. Gardner, Stephen Tyree, Kilian Q. Weinberger, and Andrew Gordon Wilson. “Exact Gaussian processes on a million data points.” arXiv preprint arXiv:1903.08114 (2019). Accepted by NeurIPS 2019 [[Link]](https://arxiv.org/abs/1903.08114)
2. Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q., & Wilson, A. G. (2018). Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration. arXiv preprint arXiv:1809.11165. Accepted by NeurIPS 2018 [[Link]](https://arxiv.org/abs/1809.11165)
3. Sun, J., Cheng, L., & Miller III, T. F. (2021). Molecular Energy Learning Using Alternative Blackbox Matrix-Matrix Multiplication Algorithm for Exact Gaussian Process. arXiv preprint arXiv:2109.09817. [[Link]](https://arxiv.org/abs/2109.09817)
