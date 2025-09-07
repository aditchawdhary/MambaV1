"""
Performance tests for MambaInference class.

Tests cover batch inference, performance monitoring, and optimization features.
"""

import pytest
import torch
import time
from unittest.mock import Mock, patch

from mamba_training.inference.mamba_inference import (
    MambaInference,
    GenerationConfig,
    BatchGenerationOutput,
    PerformanceMetrics,
    PerformanceMonitor,
    create_optimized_inference_engine
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
        tokens = [self.bos_token_id] + [hash(char) % (self.vocab_size - 10) + 10 for char in text]
        if return_tensors == 'pt':
            return torch.tensor([tokens])
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
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
def optimized_inference_engine(mock_model, mock_tokenizer, generation_config):
    """Create an optimized MambaInference engine for testing."""
    return create_optimized_inference_engine(
        model=mock_model,
        tokenizer=mock_tokenizer,
        generation_config=generation_config,
        device=torch.device('cpu'),
        max_batch_size=8,
        max_cache_size=100
    )


class TestBatchInference:
    """Test batch inference capabilities."""
    
    def test_batch_generate_basic(self, optimized_inference_engine):
        """Test basic batch generation functionality."""
        input_ids_list = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 5]]),
            torch.tensor([[6, 7, 8, 9]])
        ]
        
        config = GenerationConfig(max_length=10, do_sample=False)
        
        output = optimized_inference_engine.batch_generate(input_ids_list, config)
        
        assert isinstance(output, BatchGenerationOutput)
        assert len(output.sequences) == 3
        assert len(output.generation_times) == 3
        assert len(output.tokens_per_second) == 3
        assert output.total_time > 0
        assert output.average_tokens_per_second >= 0
        assert output.batch_efficiency > 0
    
    def test_dynamic_batching(self, optimized_inference_engine):
        """Test dynamic batching with different sequence lengths."""
        # Create inputs with varying lengths
        input_ids_list = [
            torch.tensor([[1, 2]]),
            torch.tensor([[3, 4, 5, 6, 7]]),
            torch.tensor([[8]]),
            torch.tensor([[9, 10, 11]])
        ]
        
        config = GenerationConfig(max_length=8, do_sample=False)
        
        output = optimized_inference_engine.batch_generate(input_ids_list, config)
        
        assert len(output.sequences) == 4
        # Check that sequences maintain their relative ordering
        for i, seq in enumerate(output.sequences):
            assert seq is not None
            assert seq.dim() == 1  # Should be 1D tensor
    
    def test_batch_config_management(self, optimized_inference_engine):
        """Test batch configuration management."""
        # Get initial config
        initial_config = optimized_inference_engine.get_batch_config()
        assert 'max_batch_size' in initial_config
        assert 'dynamic_batching' in initial_config
        
        # Update config
        optimized_inference_engine.set_batch_config(
            max_batch_size=16,
            dynamic_batching=False
        )
        
        updated_config = optimized_inference_engine.get_batch_config()
        assert updated_config['max_batch_size'] == 16
        assert updated_config['dynamic_batching'] is False
    
    def test_invalid_batch_config(self, optimized_inference_engine):
        """Test error handling for invalid batch configuration."""
        with pytest.raises(ValueError, match="Unknown batch config parameter"):
            optimized_inference_engine.set_batch_config(invalid_param=True)
    
    def test_empty_batch(self, optimized_inference_engine):
        """Test error handling for empty batch."""
        with pytest.raises(ValueError, match="Empty input list"):
            optimized_inference_engine._create_padded_batch([])
    
    def test_padded_batch_creation(self, optimized_inference_engine):
        """Test creation of padded batches."""
        input_ids_list = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 5]]),
            torch.tensor([[6, 7, 8, 9, 10]])
        ]
        
        batch_tensor, attention_mask = optimized_inference_engine._create_padded_batch(input_ids_list)
        
        # Check batch tensor shape
        assert batch_tensor.shape[0] == 3  # Batch size
        assert batch_tensor.shape[1] == 5  # Max sequence length
        
        # Check attention mask
        assert attention_mask.shape == batch_tensor.shape
        assert attention_mask[0, :3].all()  # First sequence has 3 tokens
        assert attention_mask[1, :2].all()  # Second sequence has 2 tokens
        assert attention_mask[2, :5].all()  # Third sequence has 5 tokens


class TestPerformanceMonitoring:
    """Test performance monitoring and metrics."""
    
    def test_performance_metrics_collection(self, optimized_inference_engine):
        """Test collection of performance metrics."""
        # Reset stats
        optimized_inference_engine.reset_stats()
        
        # Generate some data
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(max_length=8, do_sample=False)
        
        for _ in range(3):
            optimized_inference_engine.generate(input_ids, config)
        
        metrics = optimized_inference_engine.get_performance_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.throughput_tokens_per_second >= 0
        assert metrics.latency_ms_per_token >= 0
        assert metrics.memory_usage_mb >= 0
    
    def test_memory_usage_tracking(self, optimized_inference_engine):
        """Test memory usage tracking."""
        memory_usage = optimized_inference_engine._get_memory_usage()
        
        assert isinstance(memory_usage, float)
        assert memory_usage >= 0
    
    def test_cache_optimization(self, optimized_inference_engine):
        """Test cache optimization functionality."""
        # Fill cache beyond limit
        for i in range(150):  # More than max_cache_size (100)
            optimized_inference_engine._state_cache[f'key_{i}'] = f'value_{i}'
        
        # Optimize cache
        optimized_inference_engine.optimize_cache()
        
        # Check that cache size is within limit
        assert len(optimized_inference_engine._state_cache) <= optimized_inference_engine._max_cache_size
    
    def test_cache_stats_tracking(self, optimized_inference_engine):
        """Test cache statistics tracking."""
        # Simulate cache hits and misses
        optimized_inference_engine._cache_stats['hits'] = 10
        optimized_inference_engine._cache_stats['misses'] = 5
        
        metrics = optimized_inference_engine.get_performance_metrics()
        
        expected_hit_rate = 10 / (10 + 5)  # 0.667
        assert abs(metrics.cache_hit_rate - expected_hit_rate) < 0.01
    
    def test_batch_efficiency_calculation(self, optimized_inference_engine):
        """Test batch efficiency calculation."""
        # Simulate some batch generations
        optimized_inference_engine._generation_stats['batch_generations'] = 2
        optimized_inference_engine._generation_stats['total_batch_time'] = 1.0
        optimized_inference_engine._generation_stats['num_generations'] = 4
        optimized_inference_engine._generation_stats['total_generation_time'] = 3.0
        
        metrics = optimized_inference_engine.get_performance_metrics()
        
        # Batch efficiency should be calculated
        assert metrics.batch_efficiency > 0


class TestPerformanceMonitorClass:
    """Test the PerformanceMonitor class."""
    
    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()
        
        assert monitor.metrics_history == []
        assert monitor.start_time > 0
    
    def test_record_generation(self):
        """Test recording generation metrics."""
        monitor = PerformanceMonitor()
        
        monitor.record_generation(
            tokens_generated=100,
            generation_time=2.0,
            memory_usage=512.0
        )
        
        assert len(monitor.metrics_history) == 1
        
        metrics = monitor.metrics_history[0]
        assert metrics['tokens_generated'] == 100
        assert metrics['generation_time'] == 2.0
        assert metrics['memory_usage_mb'] == 512.0
        assert metrics['tokens_per_second'] == 50.0
    
    def test_average_metrics_calculation(self):
        """Test average metrics calculation."""
        monitor = PerformanceMonitor()
        
        # Record multiple generations
        monitor.record_generation(100, 2.0, 512.0)
        monitor.record_generation(200, 4.0, 600.0)
        monitor.record_generation(150, 3.0, 550.0)
        
        avg_metrics = monitor.get_average_metrics()
        
        assert avg_metrics['total_generations'] == 3
        assert avg_metrics['total_tokens'] == 450
        assert avg_metrics['total_time'] == 9.0
        assert avg_metrics['average_tokens_per_second'] == 50.0
        assert avg_metrics['average_memory_usage_mb'] == 554.0
    
    def test_time_window_filtering(self):
        """Test filtering metrics by time window."""
        monitor = PerformanceMonitor()
        
        # Mock timestamps
        with patch('time.time') as mock_time:
            # Record at different times
            mock_time.return_value = 1000.0
            monitor.record_generation(100, 1.0, 500.0)
            
            mock_time.return_value = 1005.0
            monitor.record_generation(200, 2.0, 600.0)
            
            mock_time.return_value = 1010.0
            monitor.record_generation(150, 1.5, 550.0)
            
            # Get metrics for last 7 seconds (should include last 2 entries)
            mock_time.return_value = 1012.0
            recent_metrics = monitor.get_average_metrics(window_seconds=7.0)
            
            assert recent_metrics['total_generations'] == 2
            assert recent_metrics['total_tokens'] == 350  # 200 + 150
    
    def test_history_size_limit(self):
        """Test that metrics history is limited in size."""
        monitor = PerformanceMonitor()
        
        # Record more than the limit (1000)
        for i in range(1100):
            monitor.record_generation(10, 0.1, 100.0)
        
        # Should be limited to 1000 entries
        assert len(monitor.metrics_history) == 1000


class TestOptimizedFactory:
    """Test the optimized inference engine factory."""
    
    def test_create_optimized_engine(self, mock_model, mock_tokenizer):
        """Test creation of optimized inference engine."""
        config = GenerationConfig(max_length=50)
        
        engine = create_optimized_inference_engine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            generation_config=config,
            device=torch.device('cpu'),
            max_batch_size=16,
            max_cache_size=500
        )
        
        assert isinstance(engine, MambaInference)
        assert engine._max_cache_size == 500
        
        batch_config = engine.get_batch_config()
        assert batch_config['max_batch_size'] == 16
        assert batch_config['dynamic_batching'] is True


class TestIntegrationPerformance:
    """Integration tests for performance features."""
    
    def test_end_to_end_batch_performance(self, optimized_inference_engine):
        """Test end-to-end batch performance."""
        # Create multiple inputs of varying lengths
        input_ids_list = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 5, 6, 7]]),
            torch.tensor([[8, 9]]),
            torch.tensor([[10, 11, 12, 13, 14]])
        ]
        
        config = GenerationConfig(max_length=10, do_sample=False)
        
        # Measure performance
        start_time = time.time()
        output = optimized_inference_engine.batch_generate(input_ids_list, config)
        end_time = time.time()
        
        # Verify results
        assert len(output.sequences) == 4
        assert output.total_time > 0
        assert output.batch_efficiency > 0
        
        # Check that measured time is reasonable
        actual_time = end_time - start_time
        assert abs(output.total_time - actual_time) < actual_time * 0.5
    
    def test_performance_under_load(self, optimized_inference_engine):
        """Test performance under sustained load."""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        config = GenerationConfig(max_length=8, do_sample=False)
        
        # Generate multiple times to simulate load
        for _ in range(10):
            optimized_inference_engine.generate(input_ids, config)
        
        # Check that performance metrics are reasonable
        metrics = optimized_inference_engine.get_performance_metrics()
        
        assert metrics.throughput_tokens_per_second > 0
        assert metrics.latency_ms_per_token >= 0
        assert metrics.memory_usage_mb > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_tracking(self, mock_model, mock_tokenizer):
        """Test GPU memory tracking (if CUDA available)."""
        device = torch.device('cuda')
        engine = create_optimized_inference_engine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=device
        )
        
        memory_usage = engine._get_memory_usage()
        assert memory_usage >= 0


if __name__ == "__main__":
    pytest.main([__file__])