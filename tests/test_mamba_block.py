"""
Unit tests for MambaBlock implementation.

This module contains comprehensive tests for the MambaBlock class,
including forward pass validation, gradient computation, parameter shapes,
inference caching, and integration with SelectiveSSM.
"""

import pytest
import torch
import torch.nn as nn
import math
from typing import Dict, Any

from mamba_training.models.mamba_block import MambaBlock
from mamba_training.config import MambaConfig


class TestMambaBlock:
    """Test suite for MambaBlock functionality."""
    
    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        """Default configuration for testing."""
        return {
            'd_model': 64,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'layer_idx': 0
        }
    
    @pytest.fixture
    def mamba_block(self, default_config) -> MambaBlock:
        """Create a MambaBlock instance for testing."""
        return MambaBlock(**default_config)
    
    @pytest.fixture
    def test_input(self) -> torch.Tensor:
        """Create test input tensor."""
        batch_size, seq_len, d_model = 2, 10, 64
        return torch.randn(batch_size, seq_len, d_model)
    
    def test_initialization(self, default_config):
        """Test MambaBlock initialization and parameter setup."""
        block = MambaBlock(**default_config)
        
        # Check basic attributes
        assert block.d_model == default_config['d_model']
        assert block.d_state == default_config['d_state']
        assert block.d_conv == default_config['d_conv']
        assert block.expand == default_config['expand']
        assert block.d_inner == default_config['d_model'] * default_config['expand']
        
        # Check layer existence
        assert isinstance(block.in_proj, nn.Linear)
        assert isinstance(block.conv1d, nn.Conv1d)
        assert isinstance(block.out_proj, nn.Linear)
        assert isinstance(block.norm, nn.LayerNorm)
        assert hasattr(block, 'ssm')
        assert hasattr(block, 'dt_proj_input')
        
        print("✓ MambaBlock initialization test passed")
    
    def test_parameter_shapes(self, mamba_block, default_config):
        """Test that all parameters have correct shapes."""
        d_model = default_config['d_model']
        d_inner = d_model * default_config['expand']
        d_conv = default_config['d_conv']
        dt_rank = max(d_model // 16, 1)
        
        # Input projection: d_model -> d_inner * 2
        assert mamba_block.in_proj.weight.shape == (d_inner * 2, d_model)
        
        # Convolution: depthwise conv with d_inner channels
        assert mamba_block.conv1d.weight.shape == (d_inner, 1, d_conv)
        assert mamba_block.conv1d.groups == d_inner
        
        # dt projection input: d_inner -> dt_rank
        assert mamba_block.dt_proj_input.weight.shape == (dt_rank, d_inner)
        
        # Output projection: d_inner -> d_model
        assert mamba_block.out_proj.weight.shape == (d_model, d_inner)
        
        # Layer norm: d_model
        assert mamba_block.norm.weight.shape == (d_model,)
        assert mamba_block.norm.bias.shape == (d_model,)
        
        print("✓ MambaBlock parameter shapes test passed")
    
    def test_forward_pass_shape(self, mamba_block, test_input):
        """Test forward pass output shape."""
        batch_size, seq_len, d_model = test_input.shape
        
        with torch.no_grad():
            output = mamba_block(test_input)
        
        # Output should have same shape as input
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Output should be different from input (not identity)
        assert not torch.allclose(output, test_input, atol=1e-6)
        
        print("✓ MambaBlock forward pass shape test passed")
    
    def test_forward_pass_deterministic(self, mamba_block, test_input):
        """Test that forward pass is deterministic."""
        with torch.no_grad():
            output1 = mamba_block(test_input)
            output2 = mamba_block(test_input)
        
        # Should produce identical outputs for same input
        assert torch.allclose(output1, output2, atol=1e-6)
        
        print("✓ MambaBlock deterministic forward pass test passed")
    
    def test_gradient_computation(self, mamba_block, test_input):
        """Test gradient computation through the block."""
        test_input.requires_grad_(True)
        
        output = mamba_block(test_input)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert test_input.grad is not None
        assert test_input.grad.shape == test_input.shape
        
        # Check that all parameters have gradients
        for name, param in mamba_block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"
        
        print("✓ MambaBlock gradient computation test passed")
    
    def test_residual_connection(self, default_config):
        """Test that residual connection is working."""
        # Create a block and set all weights to zero to isolate residual connection
        block = MambaBlock(**default_config)
        
        # Zero out all parameters except layer norm (which should be identity)
        with torch.no_grad():
            # Set layer norm to identity
            block.norm.weight.fill_(1.0)
            block.norm.bias.fill_(0.0)
            
            # Zero out other weights to make the block output zero (except residual)
            block.out_proj.weight.fill_(0.0)
            if block.out_proj.bias is not None:
                block.out_proj.bias.fill_(0.0)
        
        test_input = torch.randn(2, 10, default_config['d_model'])
        
        with torch.no_grad():
            output = block(test_input)
        
        # With zero output projection, output should be close to input (residual only)
        # Note: There might be small differences due to other operations
        residual_component = output - test_input
        assert torch.allclose(residual_component, torch.zeros_like(residual_component), atol=1e-3)
        
        print("✓ MambaBlock residual connection test passed")
    
    def test_layer_normalization(self, mamba_block, test_input):
        """Test layer normalization behavior."""
        # Hook to capture layer norm input and output
        norm_input = None
        norm_output = None
        
        def norm_hook(module, input, output):
            nonlocal norm_input, norm_output
            norm_input = input[0].clone()
            norm_output = output.clone()
        
        handle = mamba_block.norm.register_forward_hook(norm_hook)
        
        with torch.no_grad():
            _ = mamba_block(test_input)
        
        handle.remove()
        
        # Check that layer norm was applied
        assert norm_input is not None
        assert norm_output is not None
        
        # Check that layer norm input is the original input
        assert torch.allclose(norm_input, test_input, atol=1e-6)
        
        # Check that layer norm output has mean ~0 and std ~1 (approximately)
        mean = norm_output.mean(dim=-1)
        std = norm_output.std(dim=-1)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-2)
        
        print("✓ MambaBlock layer normalization test passed")
    
    def test_convolution_padding(self, default_config):
        """Test convolution padding behavior."""
        seq_len = 15
        batch_size = 2
        d_model = default_config['d_model']
        
        block = MambaBlock(**default_config)
        test_input = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            output = block(test_input)
        
        # Output sequence length should match input
        assert output.shape[1] == seq_len
        
        print("✓ MambaBlock convolution padding test passed")
    
    def test_inference_cache_allocation(self, mamba_block):
        """Test inference cache allocation."""
        batch_size = 4
        max_seq_len = 100
        
        cache = mamba_block.allocate_inference_cache(batch_size, max_seq_len)
        
        # Check cache structure
        assert isinstance(cache, dict)
        assert 'conv_cache' in cache
        assert 'ssm_cache' in cache
        
        # Check cache shapes
        expected_conv_shape = (batch_size, mamba_block.d_inner, mamba_block.d_conv - 1)
        expected_ssm_shape = (batch_size, mamba_block.d_state)
        
        assert cache['conv_cache'].shape == expected_conv_shape
        assert cache['ssm_cache'].shape == expected_ssm_shape
        
        print("✓ MambaBlock inference cache allocation test passed")
    
    def test_step_inference(self, mamba_block):
        """Test single step inference."""
        batch_size = 2
        d_model = mamba_block.d_model
        
        # Allocate cache
        cache = mamba_block.allocate_inference_cache(batch_size, 100)
        
        # Single token input
        single_token = torch.randn(batch_size, 1, d_model)
        
        with torch.no_grad():
            output, updated_cache = mamba_block.step(single_token, cache)
        
        # Check output shape
        assert output.shape == (batch_size, 1, d_model)
        
        # Check that cache was updated
        assert isinstance(updated_cache, dict)
        assert 'conv_cache' in updated_cache
        assert 'ssm_cache' in updated_cache
        
        print("✓ MambaBlock step inference test passed")
    
    def test_different_sequence_lengths(self, mamba_block):
        """Test with different sequence lengths."""
        batch_size = 2
        d_model = mamba_block.d_model
        
        seq_lengths = [1, 5, 10, 50, 100]
        
        for seq_len in seq_lengths:
            test_input = torch.randn(batch_size, seq_len, d_model)
            
            with torch.no_grad():
                output = mamba_block(test_input)
            
            assert output.shape == (batch_size, seq_len, d_model)
        
        print("✓ MambaBlock different sequence lengths test passed")
    
    def test_different_batch_sizes(self, mamba_block):
        """Test with different batch sizes."""
        seq_len = 10
        d_model = mamba_block.d_model
        
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, seq_len, d_model)
            
            with torch.no_grad():
                output = mamba_block(test_input)
            
            assert output.shape == (batch_size, seq_len, d_model)
        
        print("✓ MambaBlock different batch sizes test passed")
    
    def test_parameter_initialization_ranges(self, default_config):
        """Test parameter initialization is within reasonable ranges."""
        block = MambaBlock(**default_config)
        
        # Check that weights are not all zeros or ones (except layer norm which should be ones)
        for name, param in block.named_parameters():
            if 'weight' in name:
                assert not torch.allclose(param, torch.zeros_like(param))
                
                # Layer norm weights are initialized to ones, which is correct
                if 'norm.weight' not in name:
                    assert not torch.allclose(param, torch.ones_like(param))
                
                # Check that weights are within reasonable range
                assert param.abs().max() < 10.0, f"Parameter {name} has values too large"
                if 'norm.weight' not in name:  # Skip this check for layer norm
                    assert param.abs().min() < 1.0, f"Parameter {name} has all values too large"
        
        print("✓ MambaBlock parameter initialization ranges test passed")
    
    def test_integration_with_config(self):
        """Test integration with MambaConfig."""
        config = MambaConfig(
            d_model=128,
            d_state=32,
            d_conv=6,
            expand=3
        )
        
        block = MambaBlock(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand
        )
        
        # Test with config-based dimensions
        batch_size, seq_len = 2, 15
        test_input = torch.randn(batch_size, seq_len, config.d_model)
        
        with torch.no_grad():
            output = block(test_input)
        
        assert output.shape == (batch_size, seq_len, config.d_model)
        
        print("✓ MambaBlock integration with config test passed")
    
    def test_numerical_stability(self, mamba_block):
        """Test numerical stability with extreme inputs."""
        batch_size, seq_len, d_model = 2, 10, mamba_block.d_model
        
        # Test with large inputs
        large_input = torch.randn(batch_size, seq_len, d_model) * 100
        with torch.no_grad():
            output_large = mamba_block(large_input)
        
        assert torch.isfinite(output_large).all(), "Output should be finite for large inputs"
        
        # Test with small inputs
        small_input = torch.randn(batch_size, seq_len, d_model) * 0.01
        with torch.no_grad():
            output_small = mamba_block(small_input)
        
        assert torch.isfinite(output_small).all(), "Output should be finite for small inputs"
        
        print("✓ MambaBlock numerical stability test passed")


def test_mamba_block_comprehensive():
    """Run comprehensive tests for MambaBlock."""
    
    # Basic functionality test
    config = {
        'd_model': 64,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'layer_idx': 0
    }
    
    block = MambaBlock(**config)
    test_input = torch.randn(2, 10, 64)
    
    # Forward pass
    with torch.no_grad():
        output = block(test_input)
    
    assert output.shape == test_input.shape
    print("✓ Basic MambaBlock functionality test passed")
    
    # Gradient test
    test_input.requires_grad_(True)
    output = block(test_input)
    loss = output.sum()
    loss.backward()
    
    assert test_input.grad is not None
    print("✓ MambaBlock gradient test passed")
    
    # Parameter count
    total_params = sum(p.numel() for p in block.parameters())
    print(f"✓ MambaBlock total parameters: {total_params:,}")


if __name__ == "__main__":
    test_mamba_block_comprehensive()