from ..creators import randn
from ..Tensor import Tensor
from ..Generator import Generator
from ..Config import Config
from .Baselayer import BaseLayer

g = Generator().manual_seed(2147483647)

class Linear(BaseLayer):
    def __init__(self, in_features, out_features, bias=True, generator=g):
        super().__init__()
        linear_layer_count = Config.increment_linear_layer_count()
        self.name = f"Linear_{linear_layer_count}"
        self.in_features = in_features
        self.out_features = out_features
        self.weight = randn(in_features, out_features, generator=generator) / in_features**0.5
        self.weight._label = self.name + ".weight"
        if bias:
            self.bias = randn(out_features, generator=generator)
            self.bias._label = self.name + ".bias"
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight + (self.bias if self.bias is not None else 0)
        out._label = self.name + ".output"
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        params = [self.weight]
        if self.bias is not None and isinstance(self.bias, Tensor):
            params.append(self.bias)
        return params

    def __repr__(self) -> str:
        return f"{self.name}(in_features={self.in_features}, out_features={self.out_features})"