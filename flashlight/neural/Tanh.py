from ..Tensor import Tensor
from ..Config import Config
from .Baselayer import BaseLayer

class Tanh(BaseLayer):
    def __init__(self):
        super().__init__()
        self.name = f"Tanh_{Config.increment_tanh_count()}"

    def forward(self, x: Tensor) -> Tensor:
        out = x.tanh()
        out._label = self.name + ".output"
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"{self.name}"

    def __str__(self) -> str:
        return f"{self.name}"

    def parameters(self) -> list[Tensor]:
        return []