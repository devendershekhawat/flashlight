"""Additional test categories."""
import numpy as np
import pytest
from ..Tensor import Tensor
from ..creators import randn, ones, zeros, arange
from ..functions import nll_loss, softmax
from ..Config import Config
from .test_utils import assert_tensor_close

class TestLossFunctions:
    """Test loss functions."""
    
    def setup_method(self):
        """Setup before each test."""
        Config.enable_backprop = True
    
    def test_nll_loss(self):
        """Test negative log likelihood loss."""
        # Create probabilities (should sum to 1)
        logits = randn(3, 5)
        probs = softmax(logits, dim=-1)
        targets = Tensor([0, 1, 2], requires_grad=False)
        
        loss = nll_loss(probs, targets)
        
        # Loss should be a scalar
        assert loss.shape == ()
        assert loss.item() > 0
    
    def test_nll_loss_gradient(self):
        """Test gradient of NLL loss."""
        logits = randn(3, 5, requires_grad=True)
        probs = softmax(logits, dim=-1)
        targets = Tensor([0, 1, 2], requires_grad=False)
        
        loss = nll_loss(probs, targets)
        loss.backward()
        
        assert logits.grad is not None

class TestBroadcasting:
    """Test broadcasting behavior."""
    
    def test_broadcasting_addition(self):
        """Test broadcasting in addition."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = Tensor([1.0, 2.0, 3.0])
        result = a + b
        expected = np.array([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]])
        assert_tensor_close(result, expected)
    
    def test_broadcasting_multiplication(self):
        """Test broadcasting in multiplication."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([2.0, 3.0])
        result = a * b
        expected = np.array([[2.0, 6.0], [6.0, 12.0]])
        assert_tensor_close(result, expected)
    
    def test_broadcasting_gradient(self):
        """Test gradient with broadcasting."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([1.0, 2.0], requires_grad=True)
        c = (a + b).sum()
        c.backward()
        
        # a gradient should be ones
        assert_tensor_close(a.grad, np.ones_like(a.data))
        # b gradient should sum over broadcasted dimension
        assert_tensor_close(b.grad, np.array([2.0, 2.0]))

class TestShapeOperations:
    """Test shape manipulation operations."""
    
    def test_view_2d_to_1d(self):
        """Test reshaping 2D to 1D."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.view(4)
        assert b.shape == (4,)
        assert_tensor_close(b, np.array([1.0, 2.0, 3.0, 4.0]))
    
    def test_view_1d_to_2d(self):
        """Test reshaping 1D to 2D."""
        a = Tensor([1.0, 2.0, 3.0, 4.0])
        b = a.view(2, 2)
        assert b.shape == (2, 2)
        assert_tensor_close(b, np.array([[1.0, 2.0], [3.0, 4.0]]))
    
    def test_view_auto_dim(self):
        """Test automatic dimension inference."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.view(-1)
        assert b.shape == (4,)
        
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = a.view(2, -1)
        assert b.shape == (2, 3)

class TestIndexing:
    """Test indexing operations."""
    
    def test_single_index(self):
        """Test single element indexing."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a[0, 0].item() == 1.0
        assert a[1, 1].item() == 4.0
    
    def test_slice_indexing(self):
        """Test slice indexing."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = a[:, 0]
        assert_tensor_close(b, np.array([1.0, 4.0]))
        
        c = a[0, :]
        assert_tensor_close(c, np.array([1.0, 2.0, 3.0]))
    
    def test_indexing_gradient(self):
        """Test gradient through indexing."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a[0, :]
        c = b.sum()
        c.backward()
        
        expected_grad = np.zeros_like(a.data)
        expected_grad[0, :] = 1.0
        assert_tensor_close(a.grad, expected_grad)

class TestActivationFunctions:
    """Test activation functions."""
    
    def test_softmax_properties(self):
        """Test softmax properties."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        probs = a.softmax(dim=-1)
        
        # Should sum to 1 along the last dimension
        assert np.allclose(probs.data.sum(axis=-1), 1.0)
        
        # All values should be between 0 and 1
        assert np.all(probs.data >= 0)
        assert np.all(probs.data <= 1)
    
    def test_tanh_range(self):
        """Test tanh output range."""
        a = Tensor([-10.0, 0.0, 10.0])
        result = a.tanh()
        
        # Tanh output should be in [-1, 1]
        assert np.all(result.data >= -1.0)
        assert np.all(result.data <= 1.0)

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_item(self):
        """Test item extraction."""
        a = Tensor([5.0])
        assert a.item() == 5.0
    
    def test_nelement(self):
        """Test element count."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.nelement() == 4
    
    def test_shape(self):
        """Test shape property."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.shape == (2, 2)
    
    def test_detach(self):
        """Test detach operation."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = a.detach()
        
        assert b.requires_grad == False
        assert_tensor_close(b, a.data)
        
        # Detached tensor should not affect gradients
        c = b * 2.0
        c.backward()
        assert a.grad is None

class TestNumericalStability:
    """Test numerical stability."""
    
    def test_softmax_numerical_stability(self):
        """Test softmax handles large values."""
        # Large values that could cause overflow
        a = Tensor([[100.0, 101.0, 102.0]])
        probs = a.softmax(dim=-1)
        
        # Should still sum to 1
        assert np.allclose(probs.data.sum(axis=-1), 1.0)
        
        # Should not contain NaN or Inf
        assert np.all(np.isfinite(probs.data))
    
    def test_log_numerical_stability(self):
        """Test log handles edge cases."""
        # Small positive values
        a = Tensor([1e-10, 1.0, 100.0])
        result = a.log()
        
        # Should not contain NaN or Inf
        assert np.all(np.isfinite(result.data))

class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_tensor(self):
        """Test operations on empty tensor."""
        a = Tensor([])
        assert a.shape == (0,)
        assert a.sum().item() == 0.0
    
    def test_scalar_tensor(self):
        """Test scalar tensor operations."""
        a = Tensor(5.0)
        b = Tensor(3.0)
        result = a + b
        assert result.item() == 8.0
    
    def test_single_element_tensor(self):
        """Test single element tensor."""
        a = Tensor([5.0])
        b = Tensor([3.0])
        result = a * b
        assert result.item() == 15.0

