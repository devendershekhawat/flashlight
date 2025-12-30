from .Baselayer import BaseLayer
from ..Tensor import Tensor

class Relu(BaseLayer):
    def __init__(self):
        super().__init__()
        self.name = "Relu"

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        return []