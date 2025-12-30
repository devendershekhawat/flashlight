from ..Tensor import Tensor
from ..Config import Config
from ..creators import zeros, ones
from ..nograd import no_grad
from .Baselayer import BaseLayer

class BatchNorm1d(BaseLayer):
    def __init__(self, num_features, momentum=0.001, training=True):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.training = training
        self.name = f"BatchNorm1d_{Config.increment_batch_norm_1d_count()}"
        self.gamma = ones(num_features, requires_grad=True); self.gamma._label = self.name + ".gamma"
        self.beta = zeros(num_features, requires_grad=True); self.beta._label = self.name + ".beta"
        self.running_mean = zeros(1, num_features); self.running_mean._label = self.name + ".running_mean"
        self.running_std = ones(1, num_features); self.running_std._label = self.name + ".running_std"

    def forward(self, x: Tensor) -> Tensor:
      if self.training:
        xmean = x.mean(0, keepdim=True)
        xstd = x.std(0, keepdim=True)
      else:
        xmean = self.running_mean
        xstd = self.running_std
      
      xmean._label = self.name + ".xmean"
      xstd._label = self.name + ".xstd"

      normalized_numerator = x - xmean; normalized_numerator._label = self.name + ".normalized_numerator"
      normalized_denominator = xstd; normalized_denominator._label = self.name + ".normalized_denominator"
      normalized = normalized_numerator / normalized_denominator; normalized._label = self.name + ".normalized"
      out = self.gamma * normalized + self.beta; out._label = self.name + ".output"

      if self.training:
        with no_grad():
          self.running_mean = self.running_mean * (1 - self.momentum) + xmean * self.momentum
          self.running_std = self.running_std * (1 - self.momentum) + xstd * self.momentum

      return out

    def __call__(self, x: Tensor) -> Tensor:
      return self.forward(x)

    def parameters(self) -> list[Tensor]:
      return [self.gamma, self.beta]

    def __repr__(self) -> str:
      return f"{self.name}(num_features={self.num_features}, momentum={self.momentum}, training={self.training})"