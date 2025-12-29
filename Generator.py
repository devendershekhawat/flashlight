import numpy as np
from .Tensor import Tensor

class Generator:
  def __init__(self, seed=None):
    self.seed = seed
    self.rng = np.random.default_rng(seed)

  def manual_seed(self, seed):
    self.seed = seed
    self.rng = np.random.default_rng(seed)