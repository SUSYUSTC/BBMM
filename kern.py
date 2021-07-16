from . import stationary
from . import adv_kern
from .stationary import RBF
from .stationary import Matern32
from .stationary import Matern52
from .adv_kern import Derivative
from .adv_kern import FullDerivative
from . import kern_operation
from .kern_operation import ProductKernel
from . import polynomial
from .polynomial import Linear


def get_kern(name):
    return eval(name)


def get_kern_obj(data):
    return eval(data['name']).from_dict(data)
