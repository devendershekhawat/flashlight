"""Tests for neural network layers."""
import numpy as np
import pytest
from ..Tensor import Tensor
from ..creators import randn, ones, zeros
from ..neural import Linear, BatchNorm1d, Tanh
from ..Config import Config
from .test_utils import assert_tensor_close, assert_grad_close

class TestLinearLayer:
    """Test Linear layer."""
    
    def setup_method(self):
        """Setup before each test."""
        Config.enable_backprop = True
    
    def test_linear_forward(self):
        """Test forward pass of Linear layer."""
        layer = Linear(3, 2, bias=True)
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = layer(x)
        
        # Check output shape
        assert out.shape == (2, 2)
        
        # Check output is computed correctly: x @ W + b
        expected = x.data @ layer.weight.data + layer.bias.data
        assert_tensor_close(out, expected)
    
    def test_linear_no_bias(self):
        """Test Linear layer without bias."""
        layer = Linear(3, 2, bias=False)
        x = Tensor([[1.0, 2.0, 3.0]])
        out = layer(x)
        
        expected = x.data @ layer.weight.data
        assert_tensor_close(out, expected)
    
    def test_linear_gradient(self):
        """Test gradient flow through Linear layer."""
        layer = Linear(3, 2, bias=True)
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
    
    def test_linear_parameters(self):
        """Test parameter retrieval."""
        layer = Linear(3, 2, bias=True)
        params = layer.parameters()
        
        assert len(params) == 2
        assert layer.weight in params
        assert layer.bias in params
        
        layer_no_bias = Linear(3, 2, bias=False)
        params_no_bias = layer_no_bias.parameters()
        assert len(params_no_bias) == 1

class TestBatchNorm1d:
    """Test BatchNorm1d layer."""
    
    def setup_method(self):
        """Setup before each test."""
        Config.enable_backprop = True
    
    def test_batchnorm_forward_training(self):
        """Test forward pass in training mode."""
        bn = BatchNorm1d(3, training=True)
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = bn(x)
        
        # Check output shape
        assert out.shape == x.shape
        
        # In training mode, should normalize using batch statistics
        # Output should be: gamma * (x - mean) / std + beta
        x_mean = x.data.mean(axis=0, keepdims=True)
        x_std = x.data.std(axis=0, keepdims=True)
        normalized = (x.data - x_mean) / x_std
        expected = bn.gamma.data * normalized + bn.beta.data
        
        assert_tensor_close(out, expected, rtol=1e-4)
    
    def test_batchnorm_forward_eval(self):
        """Test forward pass in eval mode."""
        bn = BatchNorm1d(3, training=False)
        # Set running stats
        bn.running_mean = ones(1, 3) * 2.0
        bn.running_std = ones(1, 3) * 1.5
        
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = bn(x)
        
        # Should use running statistics
        normalized = (x.data - bn.running_mean.data) / bn.running_std.data
        expected = bn.gamma.data * normalized + bn.beta.data
        
        assert_tensor_close(out, expected, rtol=1e-4)
    
    def test_batchnorm_gradient(self):
        """Test gradient flow through BatchNorm."""
        bn = BatchNorm1d(3, training=True)
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        out = bn(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert bn.gamma.grad is not None
        assert bn.beta.grad is not None
    
    def test_batchnorm_parameters(self):
        """Test parameter retrieval."""
        bn = BatchNorm1d(3)
        params = bn.parameters()
        
        assert len(params) == 2
        assert bn.gamma in params
        assert bn.beta in params

class TestTanhLayer:
    """Test Tanh activation layer."""
    
    def setup_method(self):
        """Setup before each test."""
        Config.enable_backprop = True
    
    def test_tanh_forward(self):
        """Test forward pass of Tanh layer."""
        tanh = Tanh()
        x = Tensor([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]])
        out = tanh(x)
        
        expected = np.tanh(x.data)
        assert_tensor_close(out, expected)
    
    def test_tanh_gradient(self):
        """Test gradient flow through Tanh."""
        tanh = Tanh()
        x = Tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        out = tanh(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        # tanh'(x) = 1 - tanh(x)^2
        expected_grad = 1.0 - (out.data ** 2)
        assert_grad_close(x, expected_grad)
    
    def test_tanh_parameters(self):
        """Test parameter retrieval (should be empty)."""
        tanh = Tanh()
        params = tanh.parameters()
        assert len(params) == 0

class TestNeuralNetIntegration:
    """Test integration of multiple layers."""
    
    def setup_method(self):
        """Setup before each test."""
        Config.enable_backprop = True
    
    def test_linear_tanh_stack(self):
        """Test stacking Linear and Tanh layers."""
        from ..neural import NeuralNet
        
        def calculate_output(x):
            return x
        
        def calculate_loss(output, target):
            diff = output - target
            return (diff * diff).mean()
        
        net = NeuralNet(calculate_output=calculate_output, calculate_loss=calculate_loss)
        net.add_layer(Linear(3, 4))
        net.add_layer(Tanh())
        net.add_layer(Linear(4, 2))
        
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = Tensor([[0.0, 1.0], [1.0, 0.0]])
        
        loss = net.forward(x, y)
        
        assert loss.shape == ()
        assert loss.item() > 0
    
    def test_batchnorm_integration(self):
        """Test BatchNorm integration."""
        from ..neural import NeuralNet
        
        def calculate_output(x):
            return x
        
        def calculate_loss(output, target):
            diff = output - target
            return (diff * diff).mean()
        
        net = NeuralNet(calculate_output=calculate_output, calculate_loss=calculate_loss)
        net.add_layer(Linear(3, 4))
        net.add_layer(BatchNorm1d(4, training=True))
        net.add_layer(Tanh())
        net.add_layer(Linear(4, 2))
        
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = Tensor([[0.0, 1.0], [1.0, 0.0]])
        
        loss = net.forward(x, y)
        
        assert loss.shape == ()
        
        # Test gradient flow
        loss.backward()
        params = net.parameters()
        for p in params:
            if p.grad is not None:
                assert p.grad.data is not None

