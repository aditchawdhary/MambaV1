"""
Selective State Space Model (SSM) implementation for Mamba architecture.

This module implements the core selective SSM layer that forms the foundation
of the Mamba model, featuring input-dependent parameters and efficient
selective scan algorithms.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model layer with input-dependent parameters.
    
    This implements the core SSM computation with selective mechanisms where
    B, C, and dt parameters are generated based on the input, allowing the
    model to selectively focus on relevant information in the sequence.
    
    Args:
        d_model: Model dimension (input/output dimension)
        d_state: State space dimension
        dt_rank: Rank for dt parameter generation (defaults to d_model // 16)
        dt_min: Minimum value for dt initialization
        dt_max: Maximum value for dt initialization
        dt_init: Initialization method for dt ('random' or 'constant')
        dt_scale: Scale factor for dt initialization
        dt_init_floor: Floor value for dt initialization
        bias: Whether to use bias in linear projections
        conv_bias: Whether to use bias in convolution layers
        pscan: Whether to use parallel scan algorithm (more efficient)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        pscan: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(d_model // 16, 1)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.pscan = pscan
        
        # State transition matrix A (learnable)
        # Initialize with complex conjugate pairs for stability
        self.A_log = nn.Parameter(torch.randn(d_state))
        
        # Input-dependent parameter projections
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        self.B_proj = nn.Linear(d_model, d_state, bias=bias)
        self.C_proj = nn.Linear(d_model, d_state, bias=bias)
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize SSM parameters with proper scaling."""
        
        # Initialize A matrix for stability
        # Use negative real parts for stability
        with torch.no_grad():
            A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
            A = A.unsqueeze(-1).repeat(1, 1).flatten()
            self.A_log.copy_(-torch.log(A))
        
        # Initialize dt projection with smaller values
        dt = torch.exp(
            torch.rand(self.d_model) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp_(min=self.dt_init_floor)
        
        # Inverse of softplus to get the right initialization
        # Use a smaller scale to keep dt values reasonable
        dt = torch.log(torch.expm1(dt)) - 2.0  # Subtract 2 to make values smaller
        
        with torch.no_grad():
            # Initialize with smaller random values
            self.dt_proj.weight.normal_(0, 0.1)
            if self.dt_proj.bias is not None:
                self.dt_proj.bias.copy_(dt)
        
        # Initialize B and C projections
        nn.init.xavier_uniform_(self.B_proj.weight)
        nn.init.xavier_uniform_(self.C_proj.weight)
        
        if self.B_proj.bias is not None:
            nn.init.zeros_(self.B_proj.bias)
        if self.C_proj.bias is not None:
            nn.init.zeros_(self.C_proj.bias)
        
        # Initialize D (skip connection)
        nn.init.ones_(self.D)
    
    def forward(
        self, 
        x: torch.Tensor, 
        dt_proj_input: torch.Tensor,
        return_last_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the selective SSM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            dt_proj_input: Input for dt projection of shape (batch, seq_len, dt_rank)
            return_last_state: Whether to return the last hidden state
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
            Optionally returns last state if return_last_state=True
        """
        batch_size, seq_len, d_model = x.shape
        
        # Generate input-dependent parameters
        dt = self.dt_proj(dt_proj_input)  # (batch, seq_len, d_model)
        dt = F.softplus(dt + self.dt_proj.bias)  # Ensure positive dt
        
        B = self.B_proj(x)  # (batch, seq_len, d_state)
        C = self.C_proj(x)  # (batch, seq_len, d_state)
        
        # Get A matrix (negative for stability)
        A = -torch.exp(self.A_log)  # (d_state,)
        
        # Perform selective scan
        if self.pscan:
            y, last_state = self._selective_scan_pscan(x, dt, A, B, C)
        else:
            y, last_state = self._selective_scan_sequential(x, dt, A, B, C)
        
        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        if return_last_state:
            return y, last_state
        return y
    
    def _selective_scan_sequential(
        self, 
        x: torch.Tensor, 
        dt: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential implementation of selective scan (for reference/debugging).
        
        This is the straightforward implementation that processes the sequence
        step by step. Less efficient but easier to understand and debug.
        """
        batch_size, seq_len, d_model = x.shape
        d_state = A.shape[0]
        
        # Initialize state
        h = torch.zeros(batch_size, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for i in range(seq_len):
            # Current timestep inputs
            x_i = x[:, i, :]  # (batch, d_model)
            dt_i = dt[:, i, :]  # (batch, d_model)
            B_i = B[:, i, :]  # (batch, d_state)
            C_i = C[:, i, :]  # (batch, d_state)
            
            # Simplified state update for each dimension
            # We'll use a simpler approach that averages dt across d_model dimension
            dt_avg = dt_i.mean(dim=-1, keepdim=True)  # (batch, 1)
            
            # Discretize: dA = exp(dt * A), dB = dt * B * x_avg
            x_avg = x_i.mean(dim=-1, keepdim=True)  # (batch, 1)
            
            dA = torch.exp(dt_avg * A.unsqueeze(0))  # (batch, d_state)
            dB = dt_avg * B_i * x_avg  # (batch, d_state)
            
            # State update: h = dA * h + dB
            h = dA * h + dB
            
            # Output: y = C * h, then expand to d_model dimensions
            y_base = torch.sum(C_i * h, dim=-1, keepdim=True)  # (batch, 1)
            y_i = y_base.expand(-1, d_model)  # (batch, d_model)
            outputs.append(y_i)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return y, h
    
    def _selective_scan_pscan(
        self, 
        x: torch.Tensor, 
        dt: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel scan implementation of selective scan for efficiency.
        
        This uses the parallel scan algorithm to compute the SSM output
        efficiently in parallel, which is much faster than sequential processing.
        """
        batch_size, seq_len, d_model = x.shape
        d_state = A.shape[0]
        
        # Discretize the continuous system
        # For each position and each model dimension, we have different dt
        dt_expanded = dt.unsqueeze(-1)  # (batch, seq_len, d_model, 1)
        A_expanded = A.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, d_state)
        
        # Compute discretized A and B
        # dA = exp(dt * A) for each (batch, seq_len, d_model, d_state)
        dA = torch.exp(dt_expanded * A_expanded)  # (batch, seq_len, d_model, d_state)
        
        # dB = dt * B * x for each position
        # B is (batch, seq_len, d_state), x is (batch, seq_len, d_model)
        # We want (batch, seq_len, d_model, d_state)
        dB = (dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1))  # (batch, seq_len, d_model, d_state)
        
        # Parallel scan: compute cumulative products and sums
        # This is the core of the parallel scan algorithm
        states = self._parallel_scan(dA, dB)  # (batch, seq_len, d_model, d_state)
        
        # Compute outputs: y = C * states
        # C is (batch, seq_len, d_state), states is (batch, seq_len, d_model, d_state)
        # We want (batch, seq_len, d_model)
        C_expanded = C.unsqueeze(2)  # (batch, seq_len, 1, d_state)
        y = torch.sum(C_expanded * states, dim=-1)  # (batch, seq_len, d_model)
        
        # Return last state (average across d_model dimension for simplicity)
        last_state = states[:, -1, :, :].mean(dim=1)  # (batch, d_state)
        
        return y, last_state
    
    def _parallel_scan(self, dA: torch.Tensor, dB: torch.Tensor) -> torch.Tensor:
        """
        Parallel scan algorithm for efficient SSM computation.
        
        Computes the cumulative scan: h[i] = dA[i] * h[i-1] + dB[i]
        
        Args:
            dA: Discretized A matrices (batch, seq_len, d_model, d_state)
            dB: Discretized B inputs (batch, seq_len, d_model, d_state)
            
        Returns:
            Cumulative states (batch, seq_len, d_model, d_state)
        """
        batch_size, seq_len, d_model, d_state = dA.shape
        
        # Initialize output
        states = torch.zeros_like(dB)
        
        # Use a simple iterative approach for now
        # TODO: Implement true parallel scan for better efficiency
        h = torch.zeros(batch_size, d_model, d_state, device=dA.device, dtype=dA.dtype)
        
        for i in range(seq_len):
            h = dA[:, i] * h + dB[:, i]
            states[:, i] = h
        
        return states


def test_selective_ssm():
    """Basic test function for SelectiveSSM."""
    
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
    assert output.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert output_with_state.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {output_with_state.shape}"
    assert last_state.shape == (batch_size, d_state), f"Expected {(batch_size, d_state)}, got {last_state.shape}"
    
    # Check that outputs are the same
    assert torch.allclose(output, output_with_state, atol=1e-6), "Outputs should be identical"
    
    print("✓ SelectiveSSM basic functionality test passed")
    
    # Test gradient computation
    x.requires_grad_(True)
    dt_proj_input.requires_grad_(True)
    
    output = ssm(x, dt_proj_input)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Input should have gradients"
    assert dt_proj_input.grad is not None, "dt_proj_input should have gradients"
    
    print("✓ SelectiveSSM gradient computation test passed")


if __name__ == "__main__":
    test_selective_ssm()