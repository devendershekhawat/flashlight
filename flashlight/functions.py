from .Tensor import Tensor
from .creators import arange

def tanh(x: 'Tensor') -> 'Tensor':
  return x.tanh()

def relu(x: 'Tensor') -> 'Tensor':
  return x.relu()

def log(x: 'Tensor') -> 'Tensor':
  return x.log()

def softmax(x: 'Tensor', dim=None) -> 'Tensor':
  return x.softmax(dim=dim)

def multinomial(x: 'Tensor', num_samples: int = 1, generator=None) -> 'Tensor':
  return x.multinomial(num_samples=num_samples)

def nll_loss(output_probs: 'Tensor', target: 'Tensor') -> 'Tensor':
  likelihood = output_probs[arange(output_probs.shape[0]), target]; likelihood._label = "likelihood"
  log_likelihood = log(likelihood); log_likelihood._label = "log_likelihood"
  loss = -log_likelihood.mean(); loss._label = "nll_loss"
  return loss