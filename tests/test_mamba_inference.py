"""
Unit tests for MambaInference class.

Tests cover generation correctness, sampling strategies, and performance metrics.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
import time

from mamba_training.inference.mamba_inference import (
    MambaInference,
    GenerationConfig,
    GenerationOutput,
    create_inference_engine
)
from mamba_training.models.mamba_model import MambaModel
from mamba_training.config import MambaConfig


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
    
    def encode(self, text, return_tensors=None):
        # Simple mock encoding - convert to token ids
        tokens = [self.bos_token_id] + [hash(char) % (self.vocab_size - 10) + 10 for char in text]
        if return_tensors == 'pt':
            return torch.tensor([tokens])
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=False):
        # Simple mock decoding
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Filter special tokens if requested
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [self.pad_token_id, self.eos_token_id, self.bos_token_id]]
        
        return f"decoded_text_{len(token_ids)}_tokens"
    
    def __call__(self, texts, return_tensors=None, padding=False, truncation=False):
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = []
        max_len = 0
        
        for text in texts:
            tokens = self.encode(text)
            encoded.append(tokens)
            max_len = max(max_len, len(tokens))
        
        if padding:
            for i, tokens in enumerate(encoded):
                while len(tokens) < max_len:
                    tokens.append(self.pad_token_id)
                encoded[i] = tokens
        
        if return_tensors == 'pt':
            return {'input_ids': torch.tensor(encoded)}
        
        return {'input_ids': encoded}


@pytest.fixture
def mock_model():
    """Create a mock MambaModel for testing."""
    config = MambaConfig(
        d_model=64,
        d_state=8,
        d_conv=4,
        expand=2,
        n_layers=2,
        vocab_size=1000,
        pad_token_id=0
    )
    
    model = MambaModel(config)
    model.eval()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    return MockTokenizer()


@pytest.fixture
def generation_config():
    """Create a test generation configuration."""
    return GenerationConfig(
        max_length=20,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        pad_token_id=0,
        eos_token_id=1
    )


@pytest.fixture
def inference_engine(mock_model, mock_tokenizer, generation_config):
    """Create a MambaInference engine for testing."""
    return MambaInference(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=generation_config,
        device=torch.device('cpu')
    )


class TestGenerationConfig:
    """Test GenerationConfig validation and functionality."""
    
    def test_valid_config(self):
        """Test creating valid generation config."""
        config = GenerationConfig(
            max_length=100,
            temperature=0.8,
            top_p=0.9,
            top_k=50
        )
        assert config.max_length == 100
        assert config.temperature == 0.8
        assert config.top_p == 0.9
        assert config.top_k == 50
    
    def test_invalid_max_length(self):
        """Test validation of max_length parameter."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            GenerationConfig(max_length=0)
    
    def test_invalid_temperature(self):
        """Test validation of temperature parameter."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            GenerationConfig(temperature=0)
    
    def test_invalid_top_p(self):
        """Test validation of top_p parameter."""
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            GenerationConfig(top_p=1.5)
    
    def test_invalid_top_k(self):
        """Test validation of top_k parameter."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            GenerationConfig(top_k=0)


class TestMambaInference:
    """Test MambaInference class functionality."""
    
    def test_initialization(self, mock_model, mock_tokenizer, generation_config):
        """Test MambaInference initialization."""
        inference = MambaInference(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=generation_config,
            device=torch.device('cpu')
        )
        
        assert inference.model is mock_model
        assert inference.tokenizer is mock_tokenizer
        assert inference.config is generation_config
        assert inference.device == torch.device('cpu')
        assert not inference.model.training  # Should be in eval mode
    
    def test_generate_basic(self, inference_engine):
        """Test basic generation functionality."""
        input_ids = torch.tensor([[1, 2, 3, 4]])  # Simple input
        
        output = inference_engine.generate(input_ids)
        
        assert isinstance(output, GenerationOutput)
        assert output.sequences.shape[0] == 1  # Batch size
        assert output.sequences.shape[1] >= input_ids.shape[1]  # At least input length
        assert output.generation_time > 0
        assert output.tokens_per_second >= 0
    
    def test_generate_greedy(self, inference_engine):
        """Test greedy generation (do_sample=False)."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(
            max_length=10,
            do_sample=False,
            eos_token_id=1
        )
        
        output = inference_engine.generate(input_ids, config)
        
        assert output.sequences.shape[1] <= 10
        assert torch.all(output.sequences[:, :3] == input_ids)  # Input preserved
    
    def test_generate_sampling(self, inference_engine):
        """Test sampling generation with temperature."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(
            max_length=10,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50
        )
        
        output = inference_engine.generate(input_ids, config)
        
        assert output.sequences.shape[1] <= 10
        assert torch.all(output.sequences[:, :3] == input_ids)  # Input preserved
    
    def test_generate_with_max_new_tokens(self, inference_engine):
        """Test generation with max_new_tokens parameter."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(
            max_new_tokens=5,
            do_sample=False
        )
        
        output = inference_engine.generate(input_ids, config)
        
        expected_length = input_ids.shape[1] + 5
        assert output.sequences.shape[1] <= expected_length
    
    def test_generate_batch(self, inference_engine):
        """Test batch generation."""
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Batch size 2
        config = GenerationConfig(max_length=8, do_sample=False)
        
        output = inference_engine.generate(input_ids, config)
        
        assert output.sequences.shape[0] == 2  # Batch size preserved
        assert output.sequences.shape[1] <= 8
    
    def test_generate_with_eos(self, inference_engine):
        """Test generation stops at EOS token."""
        input_ids = torch.tensor([[2, 3, 4]])  # Start with non-EOS tokens
        config = GenerationConfig(
            max_length=20,
            do_sample=False,
            eos_token_id=1
        )
        
        # Mock the model to return EOS token
        original_forward = inference_engine.model.forward
        
        def mock_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            # Force EOS token in logits
            result.logits[:, -1, 1] = 100.0  # High logit for EOS
            result.logits[:, -1, :1] = -100.0  # Low logits for others
            result.logits[:, -1, 2:] = -100.0
            return result
        
        inference_engine.model.forward = mock_forward
        
        output = inference_engine.generate(input_ids, config)
        
        # Should stop early due to EOS
        assert output.sequences.shape[1] < 20
        
        # Restore original forward
        inference_engine.model.forward = original_forward
    
    def test_top_k_filtering(self, inference_engine):
        """Test top-k filtering functionality."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        filtered = inference_engine._apply_top_k_filtering(logits, top_k=3)
        
        # Should keep only top 3 values
        finite_count = torch.isfinite(filtered).sum().item()
        assert finite_count == 3
    
    def test_top_p_filtering(self, inference_engine):
        """Test top-p (nucleus) filtering functionality."""
        # Create logits with known probabilities
        logits = torch.tensor([[10.0, 5.0, 1.0, 0.1, 0.01]])
        
        filtered = inference_engine._apply_top_p_filtering(logits, top_p=0.9)
        
        # Should filter out low probability tokens
        finite_mask = torch.isfinite(filtered)
        assert finite_mask.sum() < logits.shape[1]
    
    def test_repetition_penalty(self, inference_engine):
        """Test repetition penalty functionality."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        generated = torch.tensor([[0, 2, 3]])  # Token 0, 2, 3 appear once each
        
        penalized = inference_engine._apply_repetition_penalty(
            logits, generated, penalty=2.0
        )
        
        # Token 0 should be penalized (at index 0)
        expected_token0_value = logits[0, 0] / 2.0  # 1.0 / 2.0 = 0.5
        assert abs(penalized[0, 0] - expected_token0_value) < 1e-6
        
        # Token 2 should be penalized (at index 2)
        expected_token2_value = logits[0, 2] / 2.0  # 3.0 / 2.0 = 1.5
        assert abs(penalized[0, 2] - expected_token2_value) < 1e-6
        
        # Token 3 should be penalized (at index 3)
        expected_token3_value = logits[0, 3] / 2.0  # 4.0 / 2.0 = 2.0
        assert abs(penalized[0, 3] - expected_token3_value) < 1e-6
        
        # Token 1 should NOT be penalized (not in generated sequence)
        assert penalized[0, 1] == logits[0, 1]
    
    def test_generate_text(self, inference_engine):
        """Test text generation from string prompt."""
        prompt = "Hello world"
        
        result = inference_engine.generate_text(prompt)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_text_no_tokenizer(self, mock_model):
        """Test text generation fails without tokenizer."""
        inference = MambaInference(model=mock_model, tokenizer=None)
        
        with pytest.raises(ValueError, match="Tokenizer is required"):
            inference.generate_text("Hello")
    
    def test_batch_generate_text(self, inference_engine):
        """Test batch text generation."""
        prompts = ["Hello", "World", "Test"]
        
        results = inference_engine.batch_generate_text(prompts)
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(text, str) for text in results)
    
    def test_generation_stats(self, inference_engine):
        """Test generation statistics tracking."""
        # Reset stats
        inference_engine.reset_stats()
        
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(max_length=8, do_sample=False)
        
        # Generate a few times
        for _ in range(3):
            inference_engine.generate(input_ids, config)
        
        stats = inference_engine.get_generation_stats()
        
        assert stats['num_generations'] == 3
        assert stats['total_tokens_generated'] > 0
        assert stats['total_generation_time'] > 0
        assert stats['average_tokens_per_second'] > 0
    
    def test_model_info(self, inference_engine):
        """Test model information retrieval."""
        info = inference_engine.get_model_info()
        
        assert 'model_config' in info
        assert 'parameter_counts' in info
        assert 'memory_footprint_mb' in info
        assert 'device' in info
        assert 'model_dtype' in info
    
    def test_config_override_with_kwargs(self, inference_engine):
        """Test generation config override with kwargs."""
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Override temperature via kwargs
        output = inference_engine.generate(
            input_ids,
            temperature=0.5,
            max_length=6
        )
        
        assert output.sequences.shape[1] <= 6
    
    def test_invalid_input_dimensions(self, inference_engine):
        """Test error handling for invalid input dimensions."""
        input_ids = torch.tensor([1, 2, 3])  # 1D instead of 2D
        
        with pytest.raises(ValueError, match="input_ids must be 2D tensor"):
            inference_engine.generate(input_ids)


class TestInferenceFactory:
    """Test inference engine factory function."""
    
    def test_create_inference_engine(self, mock_model, mock_tokenizer):
        """Test factory function for creating inference engine."""
        config = GenerationConfig(max_length=50)
        
        engine = create_inference_engine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            generation_config=config,
            device=torch.device('cpu')
        )
        
        assert isinstance(engine, MambaInference)
        assert engine.model is mock_model
        assert engine.tokenizer is mock_tokenizer
        assert engine.config is config


class TestPerformanceMetrics:
    """Test performance tracking and metrics."""
    
    def test_timing_accuracy(self, inference_engine):
        """Test that timing measurements are reasonable."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(max_length=6, do_sample=False)
        
        start_time = time.time()
        output = inference_engine.generate(input_ids, config)
        actual_time = time.time() - start_time
        
        # Measured time should be close to actual time (within 50% tolerance)
        assert abs(output.generation_time - actual_time) < actual_time * 0.5
    
    def test_tokens_per_second_calculation(self, inference_engine):
        """Test tokens per second calculation."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(max_length=8, do_sample=False)
        
        output = inference_engine.generate(input_ids, config)
        
        expected_new_tokens = output.sequences.shape[1] - input_ids.shape[1]
        expected_tps = expected_new_tokens / output.generation_time
        
        # Allow for small floating point differences
        assert abs(output.tokens_per_second - expected_tps) < 0.1


if __name__ == "__main__":
    pytest.main([__file__])