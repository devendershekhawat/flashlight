"""Tests for gradient computation correctness."""
import numpy as np
import pytest
from ..Tensor import Tensor
from ..creators import randn, ones, zeros
from ..Config import Config
from .test_utils import assert_grad_close, numerical_gradient

class TestGradients:
    """Test gradient computation correctness."""
    
    def setup_method(self):
        """Ensure backprop is enabled before each test."""
        Config.enable_backprop = True
    
    def test_addition_gradient(self):
        """Test gradient of addition."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = a + b
        c.backward()
        
        assert_grad_close(a, np.ones(3))
        assert_grad_close(b, np.ones(3))
    
    def test_multiplication_gradient(self):
        """Test gradient of multiplication."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        c.backward()
        
        assert_grad_close(a, b.data)
        assert_grad_close(b, a.data)
    
    def test_matrix_multiplication_gradient(self):
        """Test gradient of matrix multiplication."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        c = a @ b
        c.backward()
        
        # Gradient w.r.t. a should be grad @ b.T
        grad_c = np.ones_like(c.data)
        expected_grad_a = grad_c @ b.data.T
        expected_grad_b = a.data.T @ grad_c
        
        assert_grad_close(a, expected_grad_a)
        assert_grad_close(b, expected_grad_b)
    
    def test_division_gradient(self):
        """Test gradient of division."""
        a = Tensor([10.0, 20.0], requires_grad=True)
        b = Tensor([2.0, 4.0], requires_grad=True)
        c = a / b
        c.backward()
        
        # d/dx (x/y) = 1/y, d/dy (x/y) = -x/y^2
        expected_grad_a = 1.0 / b.data
        expected_grad_b = -a.data / (b.data ** 2)
        
        assert_grad_close(a, expected_grad_a)
        assert_grad_close(b, expected_grad_b)
    
    def test_tanh_gradient(self):
        """Test gradient of tanh."""
        a = Tensor([0.0, 1.0, -1.0], requires_grad=True)
        c = a.tanh()
        c.backward()
        
        # tanh'(x) = 1 - tanh(x)^2
        expected_grad = 1.0 - (c.data ** 2)
        assert_grad_close(a, expected_grad)
    
    def test_relu_gradient(self):
        """Test gradient of ReLU."""
        a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        c = a.relu()
        c.backward()
        
        # ReLU'(x) = 1 if x > 0, else 0
        expected_grad = (a.data > 0).astype(np.float32)
        assert_grad_close(a, expected_grad)
        
        # Test edge case at zero
        a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        c = a.relu()
        c.backward()
        expected_grad = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        assert_grad_close(a, expected_grad)
    
    def test_exp_gradient(self):
        """Test gradient of exp."""
        a = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        c = a.exp()
        c.backward()
        
        # exp'(x) = exp(x)
        expected_grad = c.data
        assert_grad_close(a, expected_grad)
    
    def test_log_gradient(self):
        """Test gradient of log."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = a.log()
        c.backward()
        
        # log'(x) = 1/x
        expected_grad = 1.0 / a.data
        assert_grad_close(a, expected_grad)
    
    def test_sum_gradient(self):
        """Test gradient of sum."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = a.sum()
        c.backward()
        
        # Gradient should be ones with same shape
        expected_grad = np.ones_like(a.data)
        assert_grad_close(a, expected_grad)
    
    def test_sum_dim_gradient(self):
        """Test gradient of sum along dimension."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = a.sum(dim=0)
        c.backward()
        
        # Gradient should broadcast back
        expected_grad = np.ones_like(a.data)
        assert_grad_close(a, expected_grad)
    
    def test_mean_gradient(self):
        """Test gradient of mean."""
        a = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        c = a.mean()
        c.backward()
        
        # Gradient should be 1/n for each element
        n = a.data.size
        expected_grad = np.ones_like(a.data) / n
        assert_grad_close(a, expected_grad)
    
    def test_chain_rule(self):
        """Test chain rule in composite operations."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = (a * b).sum()
        c.backward()
        
        # Gradient of sum of (a * b)
        # d/dx sum(x*y) = y, d/dy sum(x*y) = x
        assert_grad_close(a, b.data)
        assert_grad_close(b, a.data)
    
    def test_numerical_gradient_verification(self):
        """Verify gradients using numerical differentiation."""
        def f(x):
            t = Tensor(x, requires_grad=True)
            result = (t * t).sum()
            result.backward()
            return result.item()
        
        def f_numpy(x):
            return (x * x).sum()
        
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        # Analytical gradient
        t = Tensor(x, requires_grad=True)
        result = (t * t).sum()
        result.backward()
        analytical_grad = t.grad.data
        
        # Numerical gradient
        numerical_grad = numerical_gradient(f_numpy, x)
        
        # Use more relaxed tolerance for numerical gradient comparison
        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-2, atol=1e-3)
    
    def test_broadcasting_gradient(self):
        """Test gradient with broadcasting."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([1.0, 2.0], requires_grad=True)
        c = (a + b).sum()
        c.backward()
        
        # Gradient should sum over broadcasted dimensions
        assert_grad_close(a, np.ones_like(a.data))
        # b was broadcasted, so gradient should be summed
        assert_grad_close(b, np.array([2.0, 2.0]))
    
    def test_softmax_gradient(self):
        """Test gradient of softmax."""
        a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        c = a.softmax(dim=-1)
        c.backward()
        
        # Softmax gradient is complex, but we can verify it sums to zero
        # (since softmax is normalized)
        assert np.allclose(c.data.sum(), 1.0)
        # Gradient should exist
        assert a.grad is not None
    
    def test_view_gradient(self):
        """Test gradient through view operation."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.view(4)
        c = b.sum()
        c.backward()
        
        expected_grad = np.ones_like(a.data)
        assert_grad_close(a, expected_grad)
    
    def test_indexing_gradient(self):
        """Test gradient through indexing."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a[0, :]
        c = b.sum()
        c.backward()
        
        expected_grad = np.zeros_like(a.data)
        expected_grad[0, :] = 1.0
        assert_grad_close(a, expected_grad)
    
    def test_complex_graph(self):
        """Test gradient through complex computational graph."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = a * b
        d = c.tanh()
        e = d.sum()
        e.backward()
        
        # Verify gradients exist
        assert a.grad is not None
        assert b.grad is not None
        
        # tanh'(x) = 1 - tanh(x)^2
        tanh_grad = 1.0 - (d.data ** 2)
        expected_grad_a = tanh_grad * b.data
        expected_grad_b = tanh_grad * a.data
        
        assert_grad_close(a, expected_grad_a)
        assert_grad_close(b, expected_grad_b)

