"""Pytest configuration."""
import pytest
from ..Config import Config

@pytest.fixture(autouse=True)
def reset_config():
    """Reset Config state before each test."""
    Config.enable_backprop = True
    Config.linear_layer_count = 0
    Config.batch_norm_1d_count = 0
    Config.tanh_count = 0
    Config.neural_net_count = 0
    yield
    # Cleanup after test if needed

