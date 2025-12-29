import numpy as np
from .Config import Config

class Tensor:
    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: tuple = (),
        _op: str = '',
        _label: str = ''
    ):
        """
        Initializes a Flashlight Tensor object.
        
        Args:
            data: The raw data (integer, float, list, numpy array, or Flashlight Tensor).
            requires_grad: Whether this tensor requires gradient computation.
            _children: A tuple of parent Flashlight Tensors that produced this Flashlight Tensor (for the computational graph).
            _op: The operation string (e.g., '+', '*') that created this Flashlight Tensor.
            _label: An optional name for the Flashlight Tensor (useful for debugging).
        """
        # Ensure data is always a numpy array
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        elif isinstance(data, np.ndarray):
            self.data = data.copy()
        else:
            self.data = np.array(data, dtype=np.float32)
        
        # Ensure float32 dtype for consistency
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)
        
        self.requires_grad = requires_grad and Config.enable_backprop
        self._children = set(_children)
        self._op = _op
        self._label = _label
        
        # Gradients are stored as Tensors (to match PyTorch API: p.grad.data)
        self.grad: Tensor | None = None
        self._backward = lambda: None
    
    def _accumulate_grad(self, grad: np.ndarray):
        """Helper method to accumulate gradients as Tensor objects."""
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = Tensor(grad.copy(), requires_grad=False)
        else:
            self.grad.data += grad

    @property
    def shape(self):
        """Returns the shape of the tensor data."""
        return self.data.shape

    def __repr__(self):
        return f"Flashlight Tensor(data={self.data}, shape={self.data.shape}, requires_grad={self.requires_grad})"

    def numpy(self) -> np.ndarray:
        """Returns the underlying numpy array."""
        return self.data

    def item(self) -> float:
        """Extracts a scalar value from a single-element tensor."""
        return float(self.data.item())

    def nelement(self) -> int:
        """Returns the total number of elements."""
        return int(self.data.size)

    def backward(self):
        """
        Automated Backpropagation using the Chain Rule.
        
        Logic:
        1. Topological Sort: Order nodes so we process gradients from output to inputs.
        2. Set Base Gradient: The gradient of the loss with respect to itself is 1.0.
        3. Iterate & Apply Chain Rule: Run the stored _backward() functions in reverse.
        """
        # --- 1. Topological Sort ---
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # --- 2. Set Base Gradient ---
        self.grad = Tensor(np.ones_like(self.data, dtype=np.float32), requires_grad=False)
        
        # --- 3. Reverse Iteration ---
        for node in reversed(topo):
            if node._backward:
                node._backward()

    # -------------------------------------------------------------------------
    # INDEXING
    # -------------------------------------------------------------------------
    
    def _convert_index(self, idx):
        """Recursively convert Tensor indices to numpy arrays."""
        if isinstance(idx, Tensor):
            data = idx.data
            if isinstance(data, np.ndarray):
                if data.ndim == 0:
                    return int(data.item()) if np.issubdtype(data.dtype, np.integer) else data.item()
                if data.dtype.kind == 'f':
                    return data.astype(np.int64)
            return data
        elif isinstance(idx, tuple):
            return tuple(self._convert_index(i) for i in idx)
        elif isinstance(idx, list):
            return [self._convert_index(i) for i in idx]
        elif isinstance(idx, (slice, type(Ellipsis))):
            return idx
        elif idx is None:
            return None
        else:
            return idx

    def __getitem__(self, idx):
        """Indexing operation with gradient support."""
        idx_converted = self._convert_index(idx)
        # Type checker doesn't understand numpy's flexible indexing, but this is safe at runtime
        out_data = self.data[idx_converted]  # type: ignore[index]
        
        if not Config.enable_backprop or not self.requires_grad:
            return Tensor(out_data, requires_grad=False)
        
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op='slice', _label=f'{self._label}[{idx}]')
        
        def _backward():
            if out.grad is None:
                return
            grad_update = np.zeros_like(self.data, dtype=np.float32)
            # Type checker doesn't understand numpy's flexible indexing, but this is safe at runtime
            grad_update[idx_converted] += out.grad.data  # type: ignore[index]
            self._accumulate_grad(grad_update)
        
        out._backward = _backward
        return out

    def __setitem__(self, idx, value):
        """In-place assignment (no gradient tracking)."""
        idx_converted = self._convert_index(idx)
        if isinstance(value, Tensor):
            value = value.data
        # Type checker doesn't understand numpy's flexible indexing, but this is safe at runtime
        self.data[idx_converted] = value  # type: ignore[index]

    # -------------------------------------------------------------------------
    # ARITHMETIC OPERATIONS
    # -------------------------------------------------------------------------

    def __add__(self, other):
        """Forward: z = x + y, Backward: gradients pass through."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out_data = self.data + other.data
        
        if not Config.enable_backprop:
            return Tensor(out_data, requires_grad=False)
        
        requires_grad = (self.requires_grad or other.requires_grad)
        out = Tensor(out_data, requires_grad=requires_grad, _children=(self, other), _op='+', 
                    _label=f'({self._label} + {other._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            if self.requires_grad:
                self._accumulate_grad(self._unbroadcast(grad, self.data.shape))
            if other.requires_grad:
                other._accumulate_grad(other._unbroadcast(grad, other.data.shape))
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        """Forward: z = x * y, Backward: dL/dx = dL/dz * y, dL/dy = dL/dz * x."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out_data = self.data * other.data
        
        if not Config.enable_backprop:
            return Tensor(out_data, requires_grad=False)
        
        requires_grad = (self.requires_grad or other.requires_grad)
        out = Tensor(out_data, requires_grad=requires_grad, _children=(self, other), _op='*',
                    _label=f'({self._label} * {other._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            if self.requires_grad:
                grad_self = self._unbroadcast(grad * other.data, self.data.shape)
                self._accumulate_grad(grad_self)
            if other.requires_grad:
                grad_other = other._unbroadcast(grad * self.data, other.data.shape)
                other._accumulate_grad(grad_other)
        
        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Matrix multiplication: Forward: z = x @ y, Backward: dL/dx = dL/dz @ y.T, dL/dy = x.T @ dL/dz."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out_data = self.data @ other.data
        
        if not Config.enable_backprop:
            return Tensor(out_data, requires_grad=False)
        
        requires_grad = (self.requires_grad or other.requires_grad)
        out = Tensor(out_data, requires_grad=requires_grad, _children=(self, other), _op='@',
                    _label=f'({self._label} @ {other._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            if self.requires_grad:
                grad_self = grad @ other.data.T
                self._accumulate_grad(grad_self)
            if other.requires_grad:
                grad_other = self.data.T @ grad
                other._accumulate_grad(grad_other)
        
        out._backward = _backward
        return out

    def __truediv__(self, other):
        """Division: Forward: z = x / y, Backward: dL/dx = dL/dz / y, dL/dy = -dL/dz * x / y^2."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out_data = self.data / other.data
        
        if not Config.enable_backprop:
            return Tensor(out_data, requires_grad=False)
        
        requires_grad = (self.requires_grad or other.requires_grad)
        out = Tensor(out_data, requires_grad=requires_grad, _children=(self, other), _op='/',
                    _label=f'({self._label} / {other._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            if self.requires_grad:
                grad_self = self._unbroadcast(grad / other.data, self.data.shape)
                self._accumulate_grad(grad_self)
            if other.requires_grad:
                grad_other = other._unbroadcast(-grad * self.data / (other.data ** 2), other.data.shape)
                other._accumulate_grad(grad_other)
        
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        """Right division: Forward: z = y / x, Backward: dL/dx = -dL/dz * y / x^2, dL/dy = dL/dz / x."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out_data = other.data / self.data
        
        if not Config.enable_backprop:
            return Tensor(out_data, requires_grad=False)
        
        requires_grad = (self.requires_grad or other.requires_grad)
        out = Tensor(out_data, requires_grad=requires_grad, _children=(other, self), _op='/',
                    _label=f'({other._label} / {self._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            if other.requires_grad:
                grad_other = other._unbroadcast(grad / self.data, other.data.shape)
                other._accumulate_grad(grad_other)
            if self.requires_grad:
                grad_self = self._unbroadcast(-grad * other.data / (self.data ** 2), self.data.shape)
                self._accumulate_grad(grad_self)
        
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out_data = self.data - other.data
        
        if not Config.enable_backprop:
            return Tensor(out_data, requires_grad=False)
        
        requires_grad = (self.requires_grad or other.requires_grad)
        out = Tensor(out_data, requires_grad=requires_grad, _children=(self, other), _op='-',
                    _label=f'({self._label} - {other._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            if self.requires_grad:
                self._accumulate_grad(self._unbroadcast(grad, self.data.shape))
            if other.requires_grad:
                other._accumulate_grad(other._unbroadcast(-grad, other.data.shape))
        
        out._backward = _backward
        return out

    def __rsub__(self, other):
        """Right subtraction: Forward: z = y - x."""
        return other + (self * Tensor(np.ones(self.data.shape) * -1, requires_grad=self.requires_grad, _label="minusones"))

    def __neg__(self):
        """Negation: Forward: z = -x."""
        return self * Tensor(np.ones(self.data.shape) * -1, requires_grad=self.requires_grad, _label="minusones")

    def __radd__(self, other):
        """Right addition."""
        return self + other

    def __rmul__(self, other):
        """Right multiplication."""
        return self * other

    def __rmatmul__(self, other):
        """Right matrix multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return other @ self

    # -------------------------------------------------------------------------
    # UTILITY: Broadcasting Fix
    # -------------------------------------------------------------------------
    
    def _unbroadcast(self, grad: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Handles the dimensionality mismatch caused by broadcasting in NumPy.
        
        Logic:
        If we did `A + B` where A is (3, 3) and B is (1, 3), NumPy 'broadcasts' B 
        to (3, 3) virtually. 
        In backward pass, we get a (3, 3) gradient. We must SUM over the broadcasted 
        dimension to get back to B's original (1, 3) shape.
        """
        if grad.shape == target_shape:
            return grad
        
        # 1. Sum out extra leading dimensions
        ndim_grad = len(grad.shape)
        ndim_target = len(target_shape)
        
        if ndim_grad > ndim_target:
            grad = grad.sum(axis=tuple(range(ndim_grad - ndim_target)))
        
        # 2. Sum out dimensions that were 1 in the target (broadcasted dims)
        for i, dim in enumerate(target_shape):
            if dim == 1 and i < len(grad.shape):
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad

    # -------------------------------------------------------------------------
    # REDUCTION OPERATIONS
    # -------------------------------------------------------------------------

    def mean(self, dim=None, keepdim=False):
        """Compute mean along specified dimension(s)."""
        if dim is None:
            dim = tuple(range(len(self.data.shape)))
        elif isinstance(dim, int):
            dim = (dim,)
        
        out_data = self.data.mean(axis=dim, keepdims=keepdim)
        
        if not Config.enable_backprop or not self.requires_grad:
            return Tensor(out_data, requires_grad=False)
        
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op='mean',
                    _label=f'mean({self._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            if not keepdim and dim is not None:
                for d in sorted(dim, reverse=True):
                    grad = np.expand_dims(grad, axis=d)
            
            # Scale by number of elements averaged
            n_elements = np.prod([self.data.shape[d] for d in dim])
            grad = grad / n_elements
            
            # Broadcast back to original shape
            grad = np.broadcast_to(grad, self.data.shape)
            
            self._accumulate_grad(grad.astype(np.float32))
        
        out._backward = _backward
        return out

    def std(self, dim=None, keepdim=False):
        """Compute standard deviation along specified dimension(s)."""
        if dim is None:
            dim = tuple(range(len(self.data.shape)))
        elif isinstance(dim, int):
            dim = (dim,)
        
        out_data = self.data.std(axis=dim, keepdims=keepdim)
        
        if not Config.enable_backprop or not self.requires_grad:
            return Tensor(out_data, requires_grad=False)
        
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op='std',
                    _label=f'std({self._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            if not keepdim and dim is not None:
                for d in sorted(dim, reverse=True):
                    grad = np.expand_dims(grad, axis=d)
            
            # Compute mean for centering
            mean_val = self.data.mean(axis=dim, keepdims=True)
            std_val = self.data.std(axis=dim, keepdims=True)
            
            # Avoid division by zero
            safe_std = np.where(std_val == 0, 1.0, std_val)
            
            # Gradient of std
            n_elements = np.prod([self.data.shape[d] for d in dim])
            diff = self.data - mean_val
            grad_input = (diff / (safe_std * n_elements)) * grad
            
            # Broadcast back
            grad_input = np.broadcast_to(grad_input, self.data.shape)
            
            self._accumulate_grad(grad_input.astype(np.float32))
        
        out._backward = _backward
        return out

    def sum(self, dim=None, keepdim=False):
        """Compute sum along specified dimension(s)."""
        if dim is None:
            dim = tuple(range(len(self.data.shape)))
        elif isinstance(dim, int):
            dim = (dim,)
        
        out_data = self.data.sum(axis=dim, keepdims=keepdim)
        
        if not Config.enable_backprop or not self.requires_grad:
            return Tensor(out_data, requires_grad=False)
        
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op='sum',
                    _label=f'sum({self._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            if not keepdim and dim is not None:
                for d in sorted(dim, reverse=True):
                    grad = np.expand_dims(grad, axis=d)
            
            grad = np.broadcast_to(grad, self.data.shape)
            
            self._accumulate_grad(grad.astype(np.float32))
        
        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # ACTIVATION FUNCTIONS
    # -------------------------------------------------------------------------

    def tanh(self):
        """Hyperbolic tangent activation."""
        out_data = np.tanh(self.data)
        
        if not Config.enable_backprop or not self.requires_grad:
            return Tensor(out_data, requires_grad=False)
        
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op='tanh',
                    _label=f'tanh({self._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            # tanh'(x) = 1 - tanh(x)^2
            grad_input = grad * (1 - out_data ** 2)
            self._accumulate_grad(grad_input.astype(np.float32))
        
        out._backward = _backward
        return out

    def exp(self):
        """Exponential function."""
        out_data = np.exp(self.data)
        
        if not Config.enable_backprop or not self.requires_grad:
            return Tensor(out_data, requires_grad=False)
        
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op='exp',
                    _label=f'exp({self._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            # exp'(x) = exp(x)
            grad_input = grad * out_data
            self._accumulate_grad(grad_input.astype(np.float32))
        
        out._backward = _backward
        return out

    def log(self):
        """Natural logarithm."""
        out_data = np.log(self.data)
        
        if not Config.enable_backprop or not self.requires_grad:
            return Tensor(out_data, requires_grad=False)
        
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op='log',
                    _label=f'log({self._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            # log'(x) = 1/x
            grad_input = grad / self.data
            self._accumulate_grad(grad_input.astype(np.float32))
        
        out._backward = _backward
        return out

    def softmax(self, dim=None):
        """Softmax activation along specified dimension."""
        if dim is None:
            dim = -1
        
        # Numerical stability: subtract max
        max_val = self.data.max(axis=dim, keepdims=True)
        shifted = self.data - max_val
        exp_data = np.exp(shifted)
        sum_exp = exp_data.sum(axis=dim, keepdims=True)
        out_data = exp_data / sum_exp
        
        if not Config.enable_backprop or not self.requires_grad:
            return Tensor(out_data, requires_grad=False)
        
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op='softmax',
                    _label=f'softmax({self._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data
            # Softmax gradient: s_i * (g_i - sum(g_j * s_j))
            grad_sum = (grad * out_data).sum(axis=dim, keepdims=True)
            grad_input = out_data * (grad - grad_sum)
            self._accumulate_grad(grad_input.astype(np.float32))
        
        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # SHAPE OPERATIONS
    # -------------------------------------------------------------------------

    def view(self, *shape):
        """Reshape tensor (view operation)."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        
        # Handle -1 for automatic dimension inference
        total_elements = self.data.size
        shape_list = list(shape)
        if -1 in shape_list:
            idx = shape_list.index(-1)
            known_elements = 1
            for i, s in enumerate(shape_list):
                if i != idx:
                    known_elements *= s
            shape_list[idx] = total_elements // known_elements
            shape = tuple(shape_list)
        
        out_data = self.data.reshape(shape)
        
        if not Config.enable_backprop or not self.requires_grad:
            return Tensor(out_data, requires_grad=False)
        
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op='view',
                    _label=f'view{shape}({self._label})')
        
        def _backward():
            if out.grad is None:
                return
            grad = out.grad.data.reshape(self.data.shape)
            self._accumulate_grad(grad.astype(np.float32))
        
        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # SAMPLING OPERATIONS
    # -------------------------------------------------------------------------

    def multinomial(self, num_samples: int = 1):
        """
        Sample from multinomial distribution.
        Note: This operation does not support gradients (non-differentiable).
        """
        # Ensure probabilities sum to 1
        probs = self.data / (self.data.sum(axis=-1, keepdims=True) + 1e-8)
        
        # Handle 1D case
        if self.data.ndim == 1:
            sample = np.random.choice(probs.shape[-1], size=num_samples, p=probs)
            result = np.array(sample, dtype=np.int64)
            if num_samples == 1:
                result = int(result.item())
            return Tensor(result, requires_grad=False)
        
        # Handle 2D case (batch dimension)
        # For each row, sample num_samples indices
        samples = []
        for i in range(probs.shape[0]):
            sample = np.random.choice(probs.shape[-1], size=num_samples, p=probs[i])
            samples.append(sample)
        
        result = np.array(samples, dtype=np.int64)
        if num_samples == 1:
            result = result.squeeze(-1)
        
        return Tensor(result, requires_grad=False)
