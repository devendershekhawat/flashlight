import numpy as np

from .Generator import Generator
from .Tensor import Tensor

def _get_shape(args):
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return args[0]
    return args

def randn(*size, generator=None, requires_grad=True):
  if generator is None:
    generator = Generator()
  if generator.seed is not None:
    np.random.seed(generator.seed)
  return Tensor(np.random.normal(size=_get_shape(size)), requires_grad=requires_grad)

def randint(low, high=None, size=None, generator=None, requires_grad=False):
  if high is None:
    high = low
    low = 0
  if generator is None:
    generator = Generator()
  if generator.seed is not None:
    np.random.seed(generator.seed)
  if size is None:
    shape = ()
  elif isinstance(size, (tuple, list)):
    shape = tuple(size)
  else:
    shape = (size,)
  return Tensor(np.random.randint(low, high, size=shape), requires_grad=requires_grad)

def ones(*size, generator=None, requires_grad=False):
  return Tensor(np.ones(_get_shape(size)), requires_grad=requires_grad)

def zeros(*size, generator=None, requires_grad=False):
  return Tensor(np.zeros(_get_shape(size)), requires_grad=requires_grad)

def arange(start=0, end=None, step=1, requires_grad=False):
  if end is None:
    end = start
    start = 0
  return Tensor(np.arange(start, end, step), requires_grad=requires_grad)