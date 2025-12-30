from .Baselayer import BaseLayer
from ..Tensor import Tensor
from ..Config import Config
from ..creators import randint
from typing import Callable, Optional, Tuple

class NeuralNet:
    def __init__(
        self,
        calculate_output: Optional[Callable[[Tensor], Tensor]] = None,
        calculate_loss: Optional[Callable[[Tensor, Tensor], Tensor]] = None
    ):
        self.layers: list[BaseLayer] = []
        self.name = f"NeuralNet_{Config.increment_neural_net_count()}"
        self.calculate_output: Optional[Callable[[Tensor], Tensor]] = calculate_output
        self.calculate_loss: Optional[Callable[[Tensor, Tensor], Tensor]] = calculate_loss
        self.loss: Optional[Tensor] = None
        self._parameters: list[Tensor] = []
        for layer in self.layers:
            self._parameters.extend(layer.parameters())

    def set_training(self, training: bool):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = training

    def add_layer(self, layer: BaseLayer):
        self.layers.append(layer)
        params = layer.parameters()
        if params:
            self._parameters.extend(params)

    def add_additional_parameter_to_the_training(self, parameter: Tensor):
        self._parameters.append(parameter)
        parameter._label = parameter._label + "." + self.name + ".additional_parameter"
        parameter.requires_grad = True
        parameter.grad = None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self(x)
        if self.calculate_loss is not None:
            loss = self.calculate_loss(x, y)
            loss._label = self.name + ".loss"
            return loss
        else:
            raise ValueError("Model does not have a loss function, please set calculate_loss")

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.calculate_output is not None:
            x = self.calculate_output(x)
        x._label = self.name + ".output"
        return x

    def parameters(self) -> list[Tensor]:
        return self._parameters

    def train(
      self,
      x: Tensor,
      y: Tensor,
      lr: float,
      log_interval: int = 100,
      batching: bool = False,
      batch_size: int = 32
    ) -> Tensor:
      if batching:
        indices = randint(0, x.shape[0], (batch_size,))
        x_batch = x[indices]
        y_batch = y[indices]
      else:
        x_batch = x
        y_batch = y

      loss = self.forward(x_batch, y_batch)
      
      params = self.parameters()
            
      for p in self.parameters():
        if p is not None:
          p.grad = None
      
      loss.backward()
      
      for p in self.parameters():
        if p is not None and p.grad is not None:
          p.data += -lr * p.grad.data
      
      return loss

    def __repr__(self) -> str:
        layer_strs = []
        for layer in self.layers:
            layer_strs.append(f"    {repr(layer)}")
        layers_repr = "[\n" + ",\n".join(layer_strs) + "\n]"
        return f"{self.name}(layers={layers_repr})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index) -> BaseLayer:
        return self.layers[index]