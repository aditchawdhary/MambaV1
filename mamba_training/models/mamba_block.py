"""
Mamba Block implementation integrating selective SSM with convolution and projections.

This module implements the complete MambaBlock that combines the SelectiveSSM
with convolution layers, input-dependent parameter generation, residual connections,
and layer normalization to form the core building block of the Mamba architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from .selective_ssm import SelectiveSSM


class MambaBlock(nn.Module):
    """
    Complete Mamba block integrating selective SSM with convolution and projections.
    
    This block implements the full Mamba architecture component that includes:
    - Input projections and expansions
    - Convolution for local dependencies
    - Selective SSM for long-range dependencies
    - Input-dependent parameter generation for B, C, and dt
    - Residual connections and layer normalization
    - Output projections
    
    Args:
        d_model: Model dimension (input/output dimension)
        d_state: State space dimension for SSM
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        dt_rank: Rank for dt parameter generation (defaults to d_model // 16)
        dt_min: Minimum value for dt initialization
        dt_max: Maximum value for dt initialization
        dt_init: Initialization method for dt ('random' or 'constant')
        dt_scale: Scale factor for dt initialization
        dt_init_floor: Floor value for dt initialization
        bias: Whether to use bias in linear projections
        conv_bias: Whether to use bias in convolution layers
        use_fast_path: Whether to use optimized implementation paths
        layer_idx: Layer index for initialization scaling
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank or max(d_model // 16, 1)
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        # Input projection to expanded dimension
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # Convolution for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,  # Depthwise convolution
            padding=d_conv - 1,
        )
        
        # Activation function
        self.activation = nn.SiLU()  # Swish activation
        
        # Selective SSM layer
        self.ssm = SelectiveSSM(
            d_model=self.d_inner,
            d_state=d_state,
            dt_rank=self.dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            bias=bias,
            conv_bias=conv_bias,
        )
        
        # Input-dependent parameter generation
        # dt projection input comes from a separate linear layer
        self.dt_proj_input = nn.Linear(self.d_inner, self.dt_rank, bias=True)
        
        # Output projection back to model dimension
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        
        # Layer normalization (applied before the block)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize MambaBlock parameters with proper scaling."""
        
        # Initialize input projection
        # Use Xavier initialization for input projection
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        
        # Initialize convolution
        # Use He initialization for convolution weights
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)
        
        # Initialize dt projection input
        nn.init.xavier_uniform_(self.dt_proj_input.weight)
        if self.dt_proj_input.bias is not None:
            nn.init.zeros_(self.dt_proj_input.bias)
        
        # Initialize output projection with smaller scale for stability
        # Scale down by sqrt(2 * num_layers) if layer_idx is provided
        scale = 1.0
        if self.layer_idx is not None:
            scale = 1.0 / math.sqrt(2 * (self.layer_idx + 1))
        
        nn.init.xavier_uniform_(self.out_proj.weight, gain=scale)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        
        # Initialize layer norm
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        inference_params: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Forward pass of the MambaBlock.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            inference_params: Optional parameters for inference optimization
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Store residual connection
        residual = x
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Input projection to expanded dimension
        # Split into two parts: x_proj and z (gate)
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_proj, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Apply convolution for local dependencies
        # Conv1d expects (batch, channels, seq_len)
        x_conv = rearrange(x_proj, 'b l d -> b d l')  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)  # (batch, d_inner, seq_len)
        
        # Remove extra padding from convolution
        if self.d_conv > 1:
            x_conv = x_conv[..., :seq_len]  # Trim to original sequence length
        
        x_conv = rearrange(x_conv, 'b d l -> b l d')  # (batch, seq_len, d_inner)
        
        # Apply activation
        x_conv = self.activation(x_conv)
        
        # Generate input-dependent dt projection input
        dt_proj_input = self.dt_proj_input(x_conv)  # (batch, seq_len, dt_rank)
        
        # Apply selective SSM
        y = self.ssm(x_conv, dt_proj_input)  # (batch, seq_len, d_inner)
        
        # Apply gating mechanism
        # z acts as a gate for the SSM output
        z = self.activation(z)  # Apply activation to gate
        y = y * z  # Element-wise gating
        
        # Output projection back to model dimension
        output = self.out_proj(y)  # (batch, seq_len, d_model)
        
        # Add residual connection
        output = output + residual
        
        return output
    
    def allocate_inference_cache(self, batch_size: int, max_seq_len: int, dtype=None, device=None):
        """
        Allocate cache for efficient inference.
        
        This method pre-allocates memory for inference caching to avoid
        repeated memory allocations during autoregressive generation.
        
        Args:
            batch_size: Batch size for inference
            max_seq_len: Maximum sequence length
            dtype: Data type for cache tensors
            device: Device for cache tensors
            
        Returns:
            Dictionary containing allocated cache tensors
        """
        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = torch.device('cpu')
        
        # Allocate cache for convolution
        conv_cache = torch.zeros(
            batch_size, self.d_inner, self.d_conv - 1,
            dtype=dtype, device=device
        )
        
        # Allocate cache for SSM state
        ssm_cache = torch.zeros(
            batch_size, self.d_state,
            dtype=dtype, device=device
        )
        
        return {
            'conv_cache': conv_cache,
            'ssm_cache': ssm_cache,
        }
    
    def step(
        self, 
        x: torch.Tensor, 
        cache: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        Single step forward pass for inference.
        
        This method processes a single token and updates the cache,
        which is useful for autoregressive generation.
        
        Args:
            x: Input tensor of shape (batch, 1, d_model) - single token
            cache: Cache dictionary from allocate_inference_cache
            
        Returns:
            Tuple of (output, updated_cache)
        """
        batch_size, seq_len, d_model = x.shape
        assert seq_len == 1, "Step function expects single token input"
        
        # Store residual connection
        residual = x
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Input projection
        xz = self.in_proj(x)  # (batch, 1, d_inner * 2)
        x_proj, z = xz.chunk(2, dim=-1)  # Each: (batch, 1, d_inner)
        
        # Apply convolution with caching
        x_proj = x_proj.squeeze(1)  # (batch, d_inner)
        
        # Update convolution cache
        conv_cache = cache['conv_cache']  # (batch, d_inner, d_conv - 1)
        
        # Shift cache and add new input
        if self.d_conv > 1:
            # Concatenate new input with cache
            conv_input = torch.cat([conv_cache, x_proj.unsqueeze(-1)], dim=-1)  # (batch, d_inner, d_conv)
            
            # Apply convolution - conv_input is already (batch, d_inner, d_conv)
            x_conv = F.conv1d(
                conv_input,  # Already has correct shape (batch, d_inner, d_conv)
                self.conv1d.weight,
                self.conv1d.bias,
                groups=self.d_inner
            )
            
            x_conv = x_conv.squeeze(-1)  # Remove seq dim: (batch, d_inner)
            
            # Update cache (shift left and add new)
            cache['conv_cache'] = conv_input[..., 1:]  # Keep last d_conv-1 elements
        else:
            # No convolution caching needed for kernel size 1
            x_conv = F.conv1d(
                x_proj.unsqueeze(-1).unsqueeze(0),
                self.conv1d.weight,
                self.conv1d.bias,
                groups=self.d_inner
            ).squeeze(0).squeeze(-1)
        
        x_conv = x_conv.unsqueeze(1)  # (batch, 1, d_inner)
        
        # Apply activation
        x_conv = self.activation(x_conv)
        
        # Generate dt projection input
        dt_proj_input = self.dt_proj_input(x_conv)  # (batch, 1, dt_rank)
        
        # Apply selective SSM (this would need to be implemented in SSM for single step)
        # For now, use regular forward pass
        y = self.ssm(x_conv, dt_proj_input)  # (batch, 1, d_inner)
        
        # Apply gating
        z = self.activation(z)
        y = y * z
        
        # Output projection
        output = self.out_proj(y)  # (batch, 1, d_model)
        
        # Add residual connection
        output = output + residual
        
        return output, cache


def test_mamba_block():
    """Test function for MambaBlock."""
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    d_model = 64
    d_state = 16
    d_conv = 4
    expand = 2
    
    # Create model
    block = MambaBlock(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        layer_idx=0
    )
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test forward pass
    with torch.no_grad():
        output = block(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print("✓ MambaBlock forward pass test passed")
    
    # Test gradient computation
    x.requires_grad_(True)
    output = block(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Input should have gradients"
    
    print("✓ MambaBlock gradient computation test passed")
    
    # Test inference cache allocation
    cache = block.allocate_inference_cache(batch_size, seq_len)
    assert 'conv_cache' in cache, "Cache should contain conv_cache"
    assert 'ssm_cache' in cache, "Cache should contain ssm_cache"
    
    print("✓ MambaBlock cache allocation test passed")
    
    # Test single step inference
    single_token = torch.randn(batch_size, 1, d_model)
    with torch.no_grad():
        step_output, updated_cache = block.step(single_token, cache)
    
    assert step_output.shape == (batch_size, 1, d_model), f"Expected {(batch_size, 1, d_model)}, got {step_output.shape}"
    
    print("✓ MambaBlock step inference test passed")
    
    # Test parameter counting
    total_params = sum(p.numel() for p in block.parameters())
    trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    
    print(f"✓ MambaBlock parameter count: {total_params:,} total, {trainable_params:,} trainable")


if __name__ == "__main__":
    test_mamba_block()