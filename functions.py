from .Tensor import Tensor

def tanh(x: 'Tensor') -> 'Tensor':
  return x.tanh()

def log(x: 'Tensor') -> 'Tensor':
  return x.log()

def softmax(x: 'Tensor', dim=None) -> 'Tensor':
  return x.softmax(dim=dim)

def multinomial(x: 'Tensor', num_samples: int = 1) -> 'Tensor':
  return x.multinomial(num_samples=num_samples)