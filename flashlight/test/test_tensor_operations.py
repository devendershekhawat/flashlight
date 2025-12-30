"""Tests for tensor operation accuracy."""
import numpy as np
import pytest
from ..Tensor import Tensor
from ..creators import randn, ones, zeros
from .test_utils import assert_tensor_close

class TestTensorOperations:
    """Test accuracy of tensor operations against numpy."""
    
    def test_addition(self):
        """Test addition operation."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        result = a + b
        expected = np.array([5.0, 7.0, 9.0])
        assert_tensor_close(result, expected)
    
    def test_addition_broadcasting(self):
        """Test addition with broadcasting."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([1.0, 2.0])
        result = a + b
        expected = np.array([[2.0, 4.0], [4.0, 6.0]])
        assert_tensor_close(result, expected)
    
    def test_subtraction(self):
        """Test subtraction operation."""
        a = Tensor([5.0, 7.0, 9.0])
        b = Tensor([1.0, 2.0, 3.0])
        result = a - b
        expected = np.array([4.0, 5.0, 6.0])
        assert_tensor_close(result, expected)
    
    def test_multiplication(self):
        """Test element-wise multiplication."""
        a = Tensor([2.0, 3.0, 4.0])
        b = Tensor([5.0, 6.0, 7.0])
        result = a * b
        expected = np.array([10.0, 18.0, 28.0])
        assert_tensor_close(result, expected)
    
    def test_division(self):
        """Test division operation."""
        a = Tensor([10.0, 18.0, 28.0])
        b = Tensor([2.0, 3.0, 4.0])
        result = a / b
        expected = np.array([5.0, 6.0, 7.0])
        assert_tensor_close(result, expected)
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        result = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert_tensor_close(result, expected)
    
    def test_sum(self):
        """Test sum operation."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = a.sum()
        expected = np.array(10.0)
        assert_tensor_close(result, expected)
    
    def test_sum_dim(self):
        """Test sum along dimension."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = a.sum(dim=0)
        expected = np.array([4.0, 6.0])
        assert_tensor_close(result, expected)
    
    def test_mean(self):
        """Test mean operation."""
        a = Tensor([1.0, 2.0, 3.0, 4.0])
        result = a.mean()
        expected = np.array(2.5)
        assert_tensor_close(result, expected)
    
    def test_mean_dim(self):
        """Test mean along dimension."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = a.mean(dim=0)
        expected = np.array([2.0, 3.0])
        assert_tensor_close(result, expected)
    
    def test_std(self):
        """Test standard deviation."""
        a = Tensor([1.0, 2.0, 3.0, 4.0])
        result = a.std()
        expected = np.std(a.data)
        assert_tensor_close(result, expected, rtol=1e-4)
    
    def test_tanh(self):
        """Test tanh activation."""
        a = Tensor([0.0, 1.0, -1.0])
        result = a.tanh()
        expected = np.tanh(a.data)
        assert_tensor_close(result, expected)
    
    def test_relu(self):
        """Test ReLU activation."""
        a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = a.relu()
        expected = np.maximum(0, a.data)
        assert_tensor_close(result, expected)
        
        # Test with all negative values
        a = Tensor([-5.0, -3.0, -1.0])
        result = a.relu()
        expected = np.array([0.0, 0.0, 0.0])
        assert_tensor_close(result, expected)
        
        # Test with all positive values
        a = Tensor([1.0, 2.0, 3.0])
        result = a.relu()
        expected = np.array([1.0, 2.0, 3.0])
        assert_tensor_close(result, expected)
    
    def test_exp(self):
        """Test exponential function."""
        a = Tensor([0.0, 1.0, 2.0])
        result = a.exp()
        expected = np.exp(a.data)
        assert_tensor_close(result, expected)
    
    def test_log(self):
        """Test logarithm."""
        a = Tensor([1.0, 2.71828, 7.389])
        result = a.log()
        expected = np.log(a.data)
        assert_tensor_close(result, expected)
    
    def test_softmax(self):
        """Test softmax activation."""
        a = Tensor([[1.0, 2.0, 3.0]])
        result = a.softmax(dim=-1)
        expected = np.exp(a.data - a.data.max()) / np.exp(a.data - a.data.max()).sum()
        assert_tensor_close(result, expected)
    
    def test_view(self):
        """Test reshape/view operation."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = a.view(4)
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        assert_tensor_close(result, expected)
    
    def test_view_auto_dim(self):
        """Test view with automatic dimension inference."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = a.view(-1)
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        assert_tensor_close(result, expected)
    
    def test_indexing(self):
        """Test tensor indexing."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = a[0, 1]
        expected = np.array(2.0)
        assert_tensor_close(result, expected)
    
    def test_slice_indexing(self):
        """Test slice indexing."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = a[:, 0]
        expected = np.array([1.0, 4.0])
        assert_tensor_close(result, expected)
    
    def test_negation(self):
        """Test negation."""
        a = Tensor([1.0, -2.0, 3.0])
        result = -a
        expected = np.array([-1.0, 2.0, -3.0])
        assert_tensor_close(result, expected)
    
    def test_scalar_operations(self):
        """Test operations with scalars."""
        a = Tensor([1.0, 2.0, 3.0])
        result = a + 1.0
        expected = np.array([2.0, 3.0, 4.0])
        assert_tensor_close(result, expected)
        
        result = a * 2.0
        expected = np.array([2.0, 4.0, 6.0])
        assert_tensor_close(result, expected)
    
    def test_large_tensor_operations(self):
        """Test operations on larger tensors."""
        a = randn(100, 100)
        b = randn(100, 100)
        result = (a + b) * 2.0
        expected = (a.data + b.data) * 2.0
        assert_tensor_close(result, expected)

