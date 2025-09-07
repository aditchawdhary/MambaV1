"""
Integration tests for the complete MambaModel architecture.

This module contains comprehensive tests for the MambaModel including:
- Model initialization and parameter validation
- Forward pass correctness and shape validation
- Gradient computation and backpropagation
- Text generation functionality
- Parameter counting and memory estimation
- Model loading and saving
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

from mamba_training.models.mamba_model import MambaModel, MambaModelOutput, create_mamba_model, load_mamba_model
from mamba_training.config import MambaConfig


class TestMambaModel:
    """Test suite for MambaModel architecture."""
    
    @pytest.fixture
    def small_config(self):
        """Create a small configuration for testing."""
        return MambaConfig(
            d_model=64,
            d_state=8,
            d_conv=4,
            expand=2,
            n_layers=2,
            vocab_size=1000,
            pad_token_id=0
        )
    
    @pytest.fixture
    def medium_config(self):
        """Create a medium configuration for testing."""
        return MambaConfig(
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
            n_layers=4,
            vocab_size=5000,
            pad_token_id=0
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensors for testing."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        # Create random token ids (avoiding padding token 0)
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'batch_size': batch_size,
            'seq_len': seq_len
        }
    
    def test_model_initialization(self, small_config):
        """Test that MambaModel initializes correctly with proper components."""
        model = MambaModel(small_config)
        
        # Check that all components are present
        assert hasattr(model, 'embeddings'), "Model should have embeddings layer"
        assert hasattr(model, 'layers'), "Model should have MambaBlock layers"
        assert hasattr(model, 'norm_f'), "Model should have final layer norm"
        assert hasattr(model, 'lm_head'), "Model should have language modeling head"
        
        # Check layer count
        assert len(model.layers) == small_config.n_layers, f"Expected {small_config.n_layers} layers, got {len(model.layers)}"
        
        # Check embedding dimensions
        assert model.embeddings.num_embeddings == small_config.vocab_size
        assert model.embeddings.embedding_dim == small_config.d_model
        assert model.embeddings.padding_idx == small_config.pad_token_id
        
        # Check output projection dimensions
        assert model.lm_head.in_features == small_config.d_model
        assert model.lm_head.out_features == small_config.vocab_size
        
        print("✓ MambaModel initialization test passed")
    
    def test_forward_pass_shapes(self, small_config, sample_input):
        """Test that forward pass produces correct output shapes."""
        model = MambaModel(small_config)
        model.eval()
        
        with torch.no_grad():
            # Test with return_dict=True
            outputs = model(
                input_ids=sample_input['input_ids'],
                return_dict=True
            )
            
            assert isinstance(outputs, MambaModelOutput), "Output should be MambaModelOutput"
            assert outputs.logits.shape == (
                sample_input['batch_size'], 
                sample_input['seq_len'], 
                small_config.vocab_size
            ), f"Expected logits shape {(sample_input['batch_size'], sample_input['seq_len'], small_config.vocab_size)}, got {outputs.logits.shape}"
            
            # Test with return_dict=False
            outputs_tuple = model(
                input_ids=sample_input['input_ids'],
                return_dict=False
            )
            
            assert isinstance(outputs_tuple, tuple), "Output should be tuple when return_dict=False"
            assert len(outputs_tuple) == 1, "Should return only logits when output_hidden_states=False"
            assert outputs_tuple[0].shape == outputs.logits.shape, "Tuple output should match dict output"
        
        print("✓ MambaModel forward pass shapes test passed")
    
    def test_forward_pass_with_hidden_states(self, small_config, sample_input):
        """Test forward pass with hidden states output."""
        model = MambaModel(small_config)
        model.eval()
        
        with torch.no_grad():
            outputs = model(
                input_ids=sample_input['input_ids'],
                output_hidden_states=True,
                return_dict=True
            )
            
            assert outputs.hidden_states is not None, "Hidden states should be returned"
            assert len(outputs.hidden_states) == small_config.n_layers + 1, f"Expected {small_config.n_layers + 1} hidden states, got {len(outputs.hidden_states)}"
            
            # Check shapes of hidden states
            for i, hidden_state in enumerate(outputs.hidden_states):
                expected_shape = (sample_input['batch_size'], sample_input['seq_len'], small_config.d_model)
                assert hidden_state.shape == expected_shape, f"Hidden state {i} has wrong shape: {hidden_state.shape} vs {expected_shape}"
        
        print("✓ MambaModel hidden states test passed")
    
    def test_gradient_computation(self, small_config, sample_input):
        """Test that gradients are computed correctly."""
        model = MambaModel(small_config)
        model.train()
        
        # Enable gradients for input
        input_ids = sample_input['input_ids'].clone()
        
        # Forward pass
        outputs = model(input_ids, return_dict=True)
        
        # Compute loss and backward pass
        loss = outputs.logits.sum()
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
                assert not torch.isinf(param.grad).any(), f"Parameter {name} has infinite gradients"
        
        print("✓ MambaModel gradient computation test passed")
    
    def test_text_generation(self, small_config):
        """Test text generation functionality."""
        model = MambaModel(small_config)
        model.eval()
        
        # Create input prompt
        batch_size = 1
        prompt_length = 5
        input_ids = torch.randint(1, small_config.vocab_size, (batch_size, prompt_length))
        
        with torch.no_grad():
            # Test greedy generation
            generated_greedy = model.generate(
                input_ids=input_ids,
                max_length=15,
                do_sample=False
            )
            
            assert generated_greedy.shape[0] == batch_size, "Batch size should be preserved"
            assert generated_greedy.shape[1] >= prompt_length, "Generated sequence should be at least as long as input"
            assert generated_greedy.shape[1] <= 15, "Generated sequence should not exceed max_length"
            
            # Test sampling generation
            generated_sample = model.generate(
                input_ids=input_ids,
                max_length=15,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=50
            )
            
            assert generated_sample.shape[0] == batch_size, "Batch size should be preserved"
            assert generated_sample.shape[1] >= prompt_length, "Generated sequence should be at least as long as input"
            assert generated_sample.shape[1] <= 15, "Generated sequence should not exceed max_length"
        
        print("✓ MambaModel text generation test passed")
    
    def test_parameter_counting(self, small_config):
        """Test parameter counting utilities."""
        model = MambaModel(small_config)
        
        param_counts = model.count_parameters()
        
        # Check that all expected keys are present
        expected_keys = ['embeddings', 'mamba_blocks', 'final_norm', 'lm_head', 'total', 'trainable']
        for key in expected_keys:
            assert key in param_counts, f"Parameter count should include {key}"
        
        # Check that counts are positive
        for key, count in param_counts.items():
            assert count >= 0, f"Parameter count for {key} should be non-negative"
        
        # Check that total matches sum of components (accounting for tied weights)
        if model._weights_tied():
            expected_total = (
                param_counts['embeddings'] + 
                param_counts['mamba_blocks'] + 
                param_counts['final_norm']
            )
        else:
            expected_total = (
                param_counts['embeddings'] + 
                param_counts['mamba_blocks'] + 
                param_counts['final_norm'] + 
                param_counts['lm_head']
            )
        
        assert param_counts['total'] == expected_total, f"Total parameter count mismatch: {param_counts['total']} vs {expected_total}"
        
        print("✓ MambaModel parameter counting test passed")
    
    def test_memory_footprint(self, small_config):
        """Test memory footprint estimation."""
        model = MambaModel(small_config)
        
        memory_footprint = model.get_memory_footprint()
        
        # Check that all expected keys are present
        expected_keys = ['embeddings_mb', 'mamba_blocks_mb', 'final_norm_mb', 'lm_head_mb', 'total_params_mb']
        for key in expected_keys:
            assert key in memory_footprint, f"Memory footprint should include {key}"
        
        # Check that all values are non-negative
        for key, value in memory_footprint.items():
            assert value >= 0, f"Memory footprint for {key} should be non-negative"
        
        print("✓ MambaModel memory footprint test passed")
    
    def test_model_saving_and_loading(self, small_config, sample_input):
        """Test model saving and loading functionality."""
        model = MambaModel(small_config)
        
        # Get initial output for comparison
        with torch.no_grad():
            initial_output = model(sample_input['input_ids'], return_dict=True)
        
        # Save model to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'model_checkpoint.pt')
            
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': small_config
            }, checkpoint_path)
            
            # Load model
            loaded_model = load_mamba_model(checkpoint_path)
            
            # Compare outputs
            with torch.no_grad():
                loaded_output = loaded_model(sample_input['input_ids'], return_dict=True)
            
            # Check that outputs are identical
            torch.testing.assert_close(
                initial_output.logits, 
                loaded_output.logits,
                msg="Loaded model should produce identical outputs"
            )
        
        print("✓ MambaModel saving and loading test passed")
    
    def test_factory_function(self, small_config):
        """Test the create_mamba_model factory function."""
        model = create_mamba_model(small_config)
        
        assert isinstance(model, MambaModel), "Factory function should return MambaModel"
        assert model.config == small_config, "Model should use provided config"
        
        print("✓ MambaModel factory function test passed")
    
    def test_weight_tying(self, small_config):
        """Test that embedding and output projection weights are tied."""
        model = MambaModel(small_config)
        
        # Check that weights are tied by default
        assert model._weights_tied(), "Embedding and output weights should be tied by default"
        assert model.embeddings.weight is model.lm_head.weight, "Weights should be the same tensor"
        
        print("✓ MambaModel weight tying test passed")
    
    def test_inference_cache_allocation(self, small_config):
        """Test inference cache allocation for all layers."""
        model = MambaModel(small_config)
        
        batch_size = 2
        max_seq_len = 50
        
        cache = model.allocate_inference_cache(batch_size, max_seq_len)
        
        # Check that cache is created for all layers
        assert len(cache) == small_config.n_layers, f"Cache should have entries for all {small_config.n_layers} layers"
        
        for i in range(small_config.n_layers):
            layer_key = f'layer_{i}'
            assert layer_key in cache, f"Cache should have entry for {layer_key}"
            
            layer_cache = cache[layer_key]
            assert 'conv_cache' in layer_cache, f"Layer {i} cache should have conv_cache"
            assert 'ssm_cache' in layer_cache, f"Layer {i} cache should have ssm_cache"
        
        print("✓ MambaModel inference cache allocation test passed")
    
    @pytest.mark.parametrize("batch_size,seq_len", [(1, 5), (2, 10), (4, 20)])
    def test_different_input_sizes(self, small_config, batch_size, seq_len):
        """Test model with different input sizes."""
        model = MambaModel(small_config)
        model.eval()
        
        input_ids = torch.randint(1, small_config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids, return_dict=True)
        
        expected_shape = (batch_size, seq_len, small_config.vocab_size)
        assert outputs.logits.shape == expected_shape, f"Expected {expected_shape}, got {outputs.logits.shape}"
        
        print(f"✓ MambaModel test with input size ({batch_size}, {seq_len}) passed")
    
    def test_model_modes(self, small_config, sample_input):
        """Test model in training and evaluation modes."""
        model = MambaModel(small_config)
        
        # Test training mode
        model.train()
        assert model.training, "Model should be in training mode"
        
        train_output = model(sample_input['input_ids'], return_dict=True)
        assert train_output.logits.requires_grad, "Training mode output should require gradients"
        
        # Test evaluation mode
        model.eval()
        assert not model.training, "Model should be in evaluation mode"
        
        with torch.no_grad():
            eval_output = model(sample_input['input_ids'], return_dict=True)
            assert not eval_output.logits.requires_grad, "Evaluation mode output should not require gradients"
        
        print("✓ MambaModel training/evaluation modes test passed")


def test_integration_full_pipeline():
    """Integration test for the complete model pipeline."""
    
    # Create configuration
    config = MambaConfig(
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=3,
        vocab_size=2000,
        pad_token_id=0
    )
    
    # Create model
    model = create_mamba_model(config)
    
    # Create sample data
    batch_size = 2
    seq_len = 15
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    
    # Test full forward pass
    model.train()
    outputs = model(input_ids, output_hidden_states=True, return_dict=True)
    
    # Verify outputs
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    assert len(outputs.hidden_states) == config.n_layers + 1
    
    # Test backward pass
    loss = outputs.logits.sum()
    loss.backward()
    
    # Verify gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    # Test generation
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids[:1, :5],  # Use first sample, first 5 tokens
            max_length=20,
            do_sample=True,
            temperature=0.8
        )
    
    assert generated.shape[0] == 1
    assert generated.shape[1] >= 5
    assert generated.shape[1] <= 20
    
    # Test parameter counting
    param_counts = model.count_parameters()
    assert param_counts['total'] > 0
    assert param_counts['trainable'] == param_counts['total']  # All parameters should be trainable
    
    print("✓ Full integration pipeline test passed")


if __name__ == "__main__":
    # Run basic tests
    config = MambaConfig(
        d_model=64,
        d_state=8,
        d_conv=4,
        expand=2,
        n_layers=2,
        vocab_size=1000,
        pad_token_id=0
    )
    
    # Test model creation
    model = MambaModel(config)
    print(f"Created MambaModel with {model.count_parameters()['total']:,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
        print(f"Forward pass successful: {outputs.logits.shape}")
    
    # Test generation
    with torch.no_grad():
        generated = model.generate(input_ids[:1, :5], max_length=15)
        print(f"Generation successful: {generated.shape}")
    
    # Run integration test
    test_integration_full_pipeline()
    
    print("\n✓ All MambaModel tests passed!")