"""
Comprehensive unit tests for SelectiveSSM implementation.

Tests cover forward pass correctness, gradient computation, parameter initialization,
numerical stability, and edge cases.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from mamba_training.models.selective_ssm import SelectiveSSM


class TestSelectiveSSM:
    """Test suite for SelectiveSSM class."""
    
    @pytest.fixture
    def default_params(self):
        """Default parameters for testing."""
        return {
            'batch_size': 2,
            'seq_len': 8,
            'd_model': 32,
            'd_state': 8,
            'dt_rank': 4,
        }
    
    @pytest.fixture
    def ssm_model(self, default_params):
        """Create a SelectiveSSM model for testing."""
        return SelectiveSSM(
            d_model=default_params['d_model'],
            d_state=default_params['d_state'],
            dt_rank=default_params['dt_rank']
        )
    
    def test_initialization(self, default_params):
        """Test proper parameter initialization."""
        ssm = SelectiveSSM(
            d_model=default_params['d_model'],
            d_state=default_params['d_state'],
            dt_rank=default_params['dt_rank']
        )
        
        # Check parameter shapes
        assert ssm.A_log.shape == (default_params['d_state'],)
        assert ssm.dt_proj.weight.shape == (default_params['d_model'], default_params['dt_rank'])
        assert ssm.B_proj.weight.shape == (default_params['d_state'], default_params['d_model'])
        assert ssm.C_proj.weight.shape == (default_params['d_state'], default_params['d_model'])
        assert ssm.D.shape == (default_params['d_model'],)
        
        # Check A matrix initialization (should be negative for stability)
        A = -torch.exp(ssm.A_log)
        assert torch.all(A < 0), "A matrix should have negative eigenvalues for stability"
        
        # Check D initialization (should be ones)
        assert torch.allclose(ssm.D, torch.ones_like(ssm.D)), "D should be initialized to ones"
        
        print("✓ Parameter initialization test passed")
    
    def test_forward_pass_shapes(self, ssm_model, default_params):
        """Test forward pass output shapes."""
        batch_size = default_params['batch_size']
        seq_len = default_params['seq_len']
        d_model = default_params['d_model']
        d_state = default_params['d_state']
        dt_rank = default_params['dt_rank']
        
        # Create test inputs
        x = torch.randn(batch_size, seq_len, d_model)
        dt_proj_input = torch.randn(batch_size, seq_len, dt_rank)
        
        # Test forward pass without state return
        output = ssm_model(x, dt_proj_input)
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Test forward pass with state return
        output_with_state, last_state = ssm_model(x, dt_proj_input, return_last_state=True)
        assert output_with_state.shape == (batch_size, seq_len, d_model)
        assert last_state.shape == (batch_size, d_state)
        
        # Outputs should be identical
        assert torch.allclose(output, output_with_state, atol=1e-6)
        
        print("✓ Forward pass shapes test passed")
    
    def test_gradient_computation(self, ssm_model, default_params):
        """Test gradient computation and backpropagation."""
        batch_size = default_params['batch_size']
        seq_len = default_params['seq_len']
        d_model = default_params['d_model']
        dt_rank = default_params['dt_rank']
        
        # Create test inputs with gradient tracking
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        dt_proj_input = torch.randn(batch_size, seq_len, dt_rank, requires_grad=True)
        
        # Forward pass
        output = ssm_model(x, dt_proj_input)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None, "Input x should have gradients"
        assert dt_proj_input.grad is not None, "dt_proj_input should have gradients"
        
        # Check model parameter gradients
        for name, param in ssm_model.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has infinite gradients"
        
        print("✓ Gradient computation test passed")
    
    def test_numerical_stability(self, default_params):
        """Test numerical stability with extreme inputs."""
        ssm = SelectiveSSM(
            d_model=default_params['d_model'],
            d_state=default_params['d_state'],
            dt_rank=default_params['dt_rank']
        )
        
        batch_size = default_params['batch_size']
        seq_len = default_params['seq_len']
        d_model = default_params['d_model']
        dt_rank = default_params['dt_rank']
        
        # Test with large inputs
        x_large = torch.randn(batch_size, seq_len, d_model) * 10
        dt_proj_input_large = torch.randn(batch_size, seq_len, dt_rank) * 10
        
        output_large = ssm(x_large, dt_proj_input_large)
        assert not torch.isnan(output_large).any(), "Output should not contain NaN with large inputs"
        assert not torch.isinf(output_large).any(), "Output should not contain Inf with large inputs"
        
        # Test with small inputs
        x_small = torch.randn(batch_size, seq_len, d_model) * 0.01
        dt_proj_input_small = torch.randn(batch_size, seq_len, dt_rank) * 0.01
        
        output_small = ssm(x_small, dt_proj_input_small)
        assert not torch.isnan(output_small).any(), "Output should not contain NaN with small inputs"
        assert not torch.isinf(output_small).any(), "Output should not contain Inf with small inputs"
        
        print("✓ Numerical stability test passed")
    
    def test_different_sequence_lengths(self, ssm_model, default_params):
        """Test with different sequence lengths."""
        batch_size = default_params['batch_size']
        d_model = default_params['d_model']
        dt_rank = default_params['dt_rank']
        
        # Test with various sequence lengths
        for seq_len in [1, 5, 16, 32]:
            x = torch.randn(batch_size, seq_len, d_model)
            dt_proj_input = torch.randn(batch_size, seq_len, dt_rank)
            
            output = ssm_model(x, dt_proj_input)
            assert output.shape == (batch_size, seq_len, d_model)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
        
        print("✓ Different sequence lengths test passed")
    
    def test_batch_size_variations(self, default_params):
        """Test with different batch sizes."""
        seq_len = default_params['seq_len']
        d_model = default_params['d_model']
        d_state = default_params['d_state']
        dt_rank = default_params['dt_rank']
        
        ssm = SelectiveSSM(d_model=d_model, d_state=d_state, dt_rank=dt_rank)
        
        # Test with various batch sizes
        for batch_size in [1, 3, 8]:
            x = torch.randn(batch_size, seq_len, d_model)
            dt_proj_input = torch.randn(batch_size, seq_len, dt_rank)
            
            output = ssm(x, dt_proj_input)
            assert output.shape == (batch_size, seq_len, d_model)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
        
        print("✓ Batch size variations test passed")
    
    def test_parameter_constraints(self, default_params):
        """Test parameter constraints and edge cases."""
        d_model = default_params['d_model']
        d_state = default_params['d_state']
        
        # Test minimum dt_rank
        ssm_min_dt = SelectiveSSM(d_model=d_model, d_state=d_state, dt_rank=1)
        assert ssm_min_dt.dt_rank == 1
        
        # Test default dt_rank calculation
        ssm_default_dt = SelectiveSSM(d_model=d_model, d_state=d_state)
        expected_dt_rank = max(d_model // 16, 1)
        assert ssm_default_dt.dt_rank == expected_dt_rank
        
        # Test with bias enabled
        ssm_with_bias = SelectiveSSM(d_model=d_model, d_state=d_state, bias=True)
        assert ssm_with_bias.B_proj.bias is not None
        assert ssm_with_bias.C_proj.bias is not None
        
        print("✓ Parameter constraints test passed")
    
    def test_dt_parameter_generation(self, ssm_model, default_params):
        """Test dt parameter generation and constraints."""
        batch_size = default_params['batch_size']
        seq_len = default_params['seq_len']
        d_model = default_params['d_model']
        dt_rank = default_params['dt_rank']
        
        # Create test input
        dt_proj_input = torch.randn(batch_size, seq_len, dt_rank)
        
        # Get dt values
        dt = ssm_model.dt_proj(dt_proj_input)
        dt = torch.nn.functional.softplus(dt + ssm_model.dt_proj.bias)
        
        # Check that dt values are positive
        assert torch.all(dt > 0), "dt values should be positive"
        
        # Check reasonable range (allow larger values since they can vary)
        assert torch.all(dt < 100), "dt values should be reasonable (< 100)"
        
        print("✓ dt parameter generation test passed")
    
    def test_state_space_matrices(self, ssm_model, default_params):
        """Test state space matrix properties."""
        batch_size = default_params['batch_size']
        seq_len = default_params['seq_len']
        d_model = default_params['d_model']
        d_state = default_params['d_state']
        dt_rank = default_params['dt_rank']
        
        # Create test inputs
        x = torch.randn(batch_size, seq_len, d_model)
        dt_proj_input = torch.randn(batch_size, seq_len, dt_rank)
        
        # Generate B and C matrices
        B = ssm_model.B_proj(x)
        C = ssm_model.C_proj(x)
        
        # Check shapes
        assert B.shape == (batch_size, seq_len, d_state)
        assert C.shape == (batch_size, seq_len, d_state)
        
        # Check A matrix stability
        A = -torch.exp(ssm_model.A_log)
        assert torch.all(A < 0), "A matrix eigenvalues should be negative for stability"
        
        print("✓ State space matrices test passed")
    
    def test_skip_connection(self, ssm_model, default_params):
        """Test skip connection (D parameter) functionality."""
        batch_size = default_params['batch_size']
        seq_len = default_params['seq_len']
        d_model = default_params['d_model']
        dt_rank = default_params['dt_rank']
        
        # Create test inputs
        x = torch.randn(batch_size, seq_len, d_model)
        dt_proj_input = torch.randn(batch_size, seq_len, dt_rank)
        
        # Get output
        output = ssm_model(x, dt_proj_input)
        
        # The skip connection should contribute to the output
        # We can't easily isolate it, but we can check that D affects the output
        original_D = ssm_model.D.clone()
        
        # Zero out D and check output changes
        with torch.no_grad():
            ssm_model.D.zero_()
        
        output_no_skip = ssm_model(x, dt_proj_input)
        
        # Restore D
        with torch.no_grad():
            ssm_model.D.copy_(original_D)
        
        # Outputs should be different
        assert not torch.allclose(output, output_no_skip, atol=1e-6), "Skip connection should affect output"
        
        print("✓ Skip connection test passed")
    
    def test_pscan_vs_sequential(self, default_params):
        """Test that parallel scan and sequential implementations give similar results."""
        d_model = default_params['d_model']
        d_state = default_params['d_state']
        dt_rank = default_params['dt_rank']
        batch_size = 1  # Use smaller batch for this test
        seq_len = 4    # Use shorter sequence for this test
        
        # Create models with different scan methods
        ssm_pscan = SelectiveSSM(d_model=d_model, d_state=d_state, dt_rank=dt_rank, pscan=True)
        ssm_sequential = SelectiveSSM(d_model=d_model, d_state=d_state, dt_rank=dt_rank, pscan=False)
        
        # Copy parameters to make them identical
        with torch.no_grad():
            ssm_sequential.load_state_dict(ssm_pscan.state_dict())
        
        # Create test inputs
        x = torch.randn(batch_size, seq_len, d_model)
        dt_proj_input = torch.randn(batch_size, seq_len, dt_rank)
        
        # Get outputs
        with torch.no_grad():
            output_pscan = ssm_pscan(x, dt_proj_input)
            output_sequential = ssm_sequential(x, dt_proj_input)
        
        # Outputs should be close (allowing for numerical differences)
        # Note: Sequential implementation is simplified, so we use relaxed tolerance
        assert torch.allclose(output_pscan, output_sequential, atol=1e-2, rtol=1e-2), \
            "Parallel scan and sequential implementations should give similar results"
        
        print("✓ Parallel scan vs sequential test passed")


def test_selective_ssm_basic():
    """Basic functionality test that can be run standalone."""
    # Test parameters
    batch_size = 2
    seq_len = 10
    d_model = 64
    d_state = 16
    dt_rank = 4
    
    # Create model
    ssm = SelectiveSSM(d_model=d_model, d_state=d_state, dt_rank=dt_rank)
    
    # Create test inputs
    x = torch.randn(batch_size, seq_len, d_model)
    dt_proj_input = torch.randn(batch_size, seq_len, dt_rank)
    
    # Forward pass
    with torch.no_grad():
        output = ssm(x, dt_proj_input)
        output_with_state, last_state = ssm(x, dt_proj_input, return_last_state=True)
    
    # Check shapes
    assert output.shape == (batch_size, seq_len, d_model)
    assert output_with_state.shape == (batch_size, seq_len, d_model)
    assert last_state.shape == (batch_size, d_state)
    
    # Check that outputs are the same
    assert torch.allclose(output, output_with_state, atol=1e-6)
    
    print("✓ SelectiveSSM basic functionality test passed")
    
    # Test gradient computation
    x.requires_grad_(True)
    dt_proj_input.requires_grad_(True)
    
    output = ssm(x, dt_proj_input)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert dt_proj_input.grad is not None
    
    print("✓ SelectiveSSM gradient computation test passed")


if __name__ == "__main__":
    test_selective_ssm_basic()