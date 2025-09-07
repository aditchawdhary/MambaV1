"""
Complete Mamba Model implementation with embedding layer and stacked MambaBlocks.

This module implements the full Mamba architecture that combines:
- Token embedding layer
- Stacked MambaBlocks for sequence processing
- Output projection and final layer normalization
- Model initialization and parameter counting utilities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass

from .mamba_block import MambaBlock
from ..config import MambaConfig


@dataclass
class MambaModelOutput:
    """Output class for MambaModel forward pass."""
    
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    past_key_values: Optional[Dict[str, Any]] = None


class MambaModel(nn.Module):
    """
    Complete Mamba model with embedding layer and stacked MambaBlocks.
    
    This model implements the full Mamba architecture including:
    - Token embedding layer with positional information
    - Stack of MambaBlocks for sequence processing
    - Final layer normalization
    - Output projection to vocabulary
    - Model initialization and parameter utilities
    
    Args:
        config: MambaConfig containing model hyperparameters
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.pad_token_id = config.pad_token_id
        
        # Token embedding layer
        self.embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id
        )
        
        # Stack of MambaBlocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                layer_idx=i
            )
            for i in range(config.n_layers)
        ])
        
        # Final layer normalization
        self.norm_f = nn.LayerNorm(config.d_model)
        
        # Output projection to vocabulary (language modeling head)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings and output projection weights (optional but common)
        # This reduces parameters and often improves performance
        self.tie_weights()
        
        # Initialize parameters
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to residual projections
        for name, module in self.named_modules():
            if isinstance(module, MambaBlock):
                # Scale down output projection for better training stability
                nn.init.normal_(
                    module.out_proj.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layers)
                )
    
    def tie_weights(self):
        """Tie the weights of the embedding and output projection layers."""
        self.lm_head.weight = self.embeddings.weight
    
    def _init_weights(self, module):
        """Initialize model weights with appropriate scaling."""
        
        if isinstance(module, nn.Linear):
            # Use normal initialization for linear layers
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Use normal initialization for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                # Zero out padding token embedding
                nn.init.constant_(module.weight[module.padding_idx], 0.0)
        elif isinstance(module, nn.LayerNorm):
            # Standard initialization for layer norm
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def get_input_embeddings(self):
        """Get the input embedding layer."""
        return self.embeddings
    
    def set_input_embeddings(self, new_embeddings):
        """Set the input embedding layer."""
        self.embeddings = new_embeddings
    
    def get_output_embeddings(self):
        """Get the output projection layer."""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set the output projection layer."""
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tuple, MambaModelOutput]:
        """
        Forward pass of the complete Mamba model.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            attention_mask: Mask to avoid attention on padding tokens (currently unused in Mamba)
            labels: Labels for language modeling loss computation
            output_hidden_states: Whether to return hidden states from all layers
            return_dict: Whether to return MambaModelOutput or tuple
            
        Returns:
            MambaModelOutput or tuple containing logits and optional hidden states
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.embeddings(input_ids)  # (batch_size, seq_len, d_model)
        
        # Store hidden states from all layers if requested
        all_hidden_states = [] if output_hidden_states else None
        
        # Pass through MambaBlocks
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            hidden_states = layer(hidden_states)
        
        # Final layer normalization
        hidden_states = self.norm_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Output projection to vocabulary
        logits = self.lm_head(hidden_states)  # (batch_size, seq_len, vocab_size)
        
        # Prepare output
        if not return_dict:
            outputs = (logits,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            return outputs
        
        return MambaModelOutput(
            logits=logits,
            hidden_states=all_hidden_states if output_hidden_states else None
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the Mamba model.
        
        Args:
            input_ids: Input token ids of shape (batch_size, seq_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token id
            eos_token_id: End-of-sequence token id
            
        Returns:
            Generated token ids of shape (batch_size, generated_length)
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids.clone()
        
        # Generate tokens one by one
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(generated, return_dict=True)
                logits = outputs.logits[:, -1, :]  # Get last token logits
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                    logits = logits_filtered
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for end-of-sequence
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters with detailed breakdown.
        
        Returns:
            Dictionary containing parameter counts for different components
        """
        param_counts = {}
        
        # Embedding parameters
        param_counts['embeddings'] = sum(p.numel() for p in self.embeddings.parameters())
        
        # MambaBlock parameters
        param_counts['mamba_blocks'] = sum(p.numel() for p in self.layers.parameters())
        
        # Final layer norm parameters
        param_counts['final_norm'] = sum(p.numel() for p in self.norm_f.parameters())
        
        # Output projection parameters (if not tied)
        if not self._weights_tied():
            param_counts['lm_head'] = sum(p.numel() for p in self.lm_head.parameters())
        else:
            param_counts['lm_head'] = 0  # Tied with embeddings
        
        # Total parameters
        param_counts['total'] = sum(p.numel() for p in self.parameters())
        param_counts['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return param_counts
    
    def _weights_tied(self) -> bool:
        """Check if embedding and output projection weights are tied."""
        return self.embeddings.weight is self.lm_head.weight
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """
        Estimate model memory footprint in MB.
        
        Returns:
            Dictionary containing memory estimates for different components
        """
        def get_param_memory(module):
            """Get memory usage of module parameters in MB."""
            return sum(p.numel() * p.element_size() for p in module.parameters()) / (1024 ** 2)
        
        memory_footprint = {}
        
        # Parameter memory
        memory_footprint['embeddings_mb'] = get_param_memory(self.embeddings)
        memory_footprint['mamba_blocks_mb'] = get_param_memory(self.layers)
        memory_footprint['final_norm_mb'] = get_param_memory(self.norm_f)
        
        if not self._weights_tied():
            memory_footprint['lm_head_mb'] = get_param_memory(self.lm_head)
        else:
            memory_footprint['lm_head_mb'] = 0.0
        
        memory_footprint['total_params_mb'] = sum([
            memory_footprint['embeddings_mb'],
            memory_footprint['mamba_blocks_mb'],
            memory_footprint['final_norm_mb'],
            memory_footprint['lm_head_mb']
        ])
        
        return memory_footprint
    
    def allocate_inference_cache(self, batch_size: int, max_seq_len: int, dtype=None, device=None):
        """
        Allocate inference cache for all layers.
        
        Args:
            batch_size: Batch size for inference
            max_seq_len: Maximum sequence length
            dtype: Data type for cache tensors
            device: Device for cache tensors
            
        Returns:
            Dictionary containing cache for all layers
        """
        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = next(self.parameters()).device
        
        cache = {}
        for i, layer in enumerate(self.layers):
            cache[f'layer_{i}'] = layer.allocate_inference_cache(
                batch_size, max_seq_len, dtype, device
            )
        
        return cache


def create_mamba_model(config: MambaConfig) -> MambaModel:
    """
    Factory function to create a MambaModel with the given configuration.
    
    Args:
        config: MambaConfig containing model hyperparameters
        
    Returns:
        Initialized MambaModel
    """
    return MambaModel(config)


def load_mamba_model(checkpoint_path: str, config: Optional[MambaConfig] = None) -> MambaModel:
    """
    Load a MambaModel from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        config: Optional config override (uses config from checkpoint if None)
        
    Returns:
        Loaded MambaModel
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Use config from checkpoint if not provided
    if config is None:
        config = checkpoint.get('config')
        if config is None:
            raise ValueError("No config found in checkpoint and none provided")
    
    # Create model and load state dict
    model = MambaModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model