"""Utility functions for testing."""
import numpy as np
from ..Tensor import Tensor

def numerical_gradient(f, x, h=1e-5):
    """
    Compute numerical gradient using finite differences.
    
    Args:
        f: Function that takes a numpy array and returns a scalar
        x: Input numpy array
        h: Step size for finite differences
    
    Returns:
        Numerical gradient with same shape as x
    """
    grad = np.zeros_like(x)
    flat_x = x.flatten()
    flat_grad = grad.flatten()
    
    for i in range(len(flat_x)):
        x_plus = flat_x.copy()
        x_plus[i] += h
        x_minus = flat_x.copy()
        x_minus[i] -= h
        
        f_plus = f(x_plus.reshape(x.shape))
        f_minus = f(x_minus.reshape(x.shape))
        
        flat_grad[i] = (f_plus - f_minus) / (2 * h)
    
    return flat_grad.reshape(x.shape)

def assert_tensor_close(t1: Tensor, t2, rtol=1e-5, atol=1e-8):
    """Assert that two tensors are close (handles both Tensor and numpy arrays)."""
    if isinstance(t2, Tensor):
        t2_data = t2.data
    else:
        t2_data = np.array(t2)
    
    np.testing.assert_allclose(t1.data, t2_data, rtol=rtol, atol=atol)

def assert_grad_close(tensor: Tensor, expected_grad, rtol=1e-4, atol=1e-6):
    """Assert that tensor gradient matches expected gradient."""
    if tensor.grad is None:
        raise AssertionError("Tensor gradient is None")
    
    if isinstance(expected_grad, Tensor):
        expected_data = expected_grad.data
    else:
        expected_data = np.array(expected_grad)
    
    np.testing.assert_allclose(tensor.grad.data, expected_data, rtol=rtol, atol=atol)

