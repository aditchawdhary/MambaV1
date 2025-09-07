"""Tests for memory optimization utilities."""

import pytest
import torch
import torch.nn as nn
import gc
import time
from unittest.mock import Mock, patch, MagicMock

from mamba_training.training.memory_optimization import (
    MemoryProfiler,
    GradientCheckpointing,
    DynamicBatchSizer,
    MemoryOptimizer,
    memory_efficient_forward,
    create_memory_optimizer
)
from mamba_training.config import TrainingConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)


class MockMambaBlock(nn.Module):
    """Mock MambaBlock for testing gradient checkpointing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def memory_profiler():
    """Create memory profiler for testing."""
    return MemoryProfiler()


@pytest.fixture
def training_config():
    """Create training configuration for testing."""
    return TrainingConfig(
        batch_size=32,
        gradient_checkpointing=True,
        use_mixed_precision=True
    )


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return SimpleModel()


class TestMemoryProfiler:
    """Test MemoryProfiler functionality."""
    
    def test_initialization(self):
        """Test MemoryProfiler initialization."""
        profiler = MemoryProfiler()
        
        assert profiler.device is not None
        assert isinstance(profiler.memory_history, list)
        assert len(profiler.memory_history) == 0
    
    def test_get_memory_stats(self, memory_profiler):
        """Test memory statistics collection."""
        stats = memory_profiler.get_memory_stats()
        
        assert 'system' in stats
        assert 'process' in stats
        
        # Check system memory stats
        system = stats['system']
        assert 'total' in system
        assert 'available' in system
        assert 'used' in system
        assert 'percent' in system
        
        # Check process memory stats
        process = stats['process']
        assert 'rss' in process
        assert 'vms' in process
        
        # All values should be positive
        assert system['total'] > 0
        assert system['used'] >= 0
        assert process['rss'] > 0
    
    def test_record_memory_snapshot(self, memory_profiler):
        """Test memory snapshot recording."""
        tag = "test_snapshot"
        memory_profiler.record_memory_snapshot(tag)
        
        assert len(memory_profiler.memory_history) == 1
        snapshot = memory_profiler.memory_history[0]
        
        assert snapshot['tag'] == tag
        assert 'timestamp' in snapshot
        assert 'system' in snapshot
        assert 'process' in snapshot
    
    def test_get_memory_diff(self, memory_profiler):
        """Test memory difference calculation."""
        # Record two snapshots
        memory_profiler.record_memory_snapshot("start")
        
        # Allocate some memory
        dummy_tensor = torch.randn(1000, 1000)
        
        memory_profiler.record_memory_snapshot("end")
        
        # Get difference
        diff = memory_profiler.get_memory_diff("start", "end")
        
        assert 'system' in diff
        assert 'process' in diff
        assert 'time_diff' in diff
        
        assert diff['time_diff'] >= 0
        
        # Clean up
        del dummy_tensor
    
    def test_get_memory_diff_missing_tags(self, memory_profiler):
        """Test memory difference with missing tags."""
        with pytest.raises(ValueError):
            memory_profiler.get_memory_diff("nonexistent_start", "nonexistent_end")
    
    def test_clear_history(self, memory_profiler):
        """Test clearing memory history."""
        memory_profiler.record_memory_snapshot("test")
        assert len(memory_profiler.memory_history) == 1
        
        memory_profiler.clear_history()
        assert len(memory_profiler.memory_history) == 0
    
    def test_profile_memory_context_manager(self, memory_profiler):
        """Test memory profiling context manager."""
        tag = "test_context"
        
        with memory_profiler.profile_memory(tag):
            # Allocate some memory
            dummy_tensor = torch.randn(100, 100)
        
        # Check that snapshots were recorded
        start_tag = f"{tag}_start"
        end_tag = f"{tag}_end"
        
        tags = [snapshot['tag'] for snapshot in memory_profiler.memory_history]
        assert start_tag in tags
        assert end_tag in tags
        
        # Clean up
        del dummy_tensor


class TestGradientCheckpointing:
    """Test GradientCheckpointing functionality."""
    
    def test_enable_gradient_checkpointing(self):
        """Test enabling gradient checkpointing."""
        model = nn.Sequential(
            MockMambaBlock(),
            nn.Linear(10, 5)
        )
        
        GradientCheckpointing.enable_gradient_checkpointing(model)
        
        # Check that MambaBlock has gradient checkpointing enabled
        mamba_block = model[0]
        assert hasattr(mamba_block, 'gradient_checkpointing')
        assert mamba_block.gradient_checkpointing is True
    
    def test_disable_gradient_checkpointing(self):
        """Test disabling gradient checkpointing."""
        model = MockMambaBlock()
        
        # First enable it
        GradientCheckpointing.enable_gradient_checkpointing(model)
        assert model.gradient_checkpointing is True
        
        # Then disable it
        GradientCheckpointing.disable_gradient_checkpointing(model)
        assert model.gradient_checkpointing is False
    
    def test_checkpointed_forward_pass(self):
        """Test that checkpointed forward pass works."""
        model = MockMambaBlock()
        GradientCheckpointing.enable_gradient_checkpointing(model)
        
        x = torch.randn(5, 10, requires_grad=True)
        output = model(x)
        
        assert output.shape == (5, 10)
        assert output.requires_grad
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestDynamicBatchSizer:
    """Test DynamicBatchSizer functionality."""
    
    def test_initialization(self):
        """Test DynamicBatchSizer initialization."""
        initial_batch_size = 32
        batcher = DynamicBatchSizer(initial_batch_size)
        
        assert batcher.current_batch_size == initial_batch_size
        assert batcher.min_batch_size == 1
        assert batcher.max_batch_size == initial_batch_size * 4
        assert batcher.successful_steps == 0
        assert batcher.oom_count == 0
    
    def test_handle_oom(self):
        """Test OOM handling."""
        batcher = DynamicBatchSizer(32, reduction_factor=0.5)
        
        original_batch_size = batcher.current_batch_size
        new_batch_size = batcher.handle_oom()
        
        assert new_batch_size == int(original_batch_size * 0.5)
        assert batcher.oom_count == 1
        assert batcher.successful_steps == 0
    
    def test_handle_oom_minimum_batch_size(self):
        """Test OOM handling with minimum batch size constraint."""
        batcher = DynamicBatchSizer(2, min_batch_size=1, reduction_factor=0.5)
        
        # First OOM should reduce to 1
        new_batch_size = batcher.handle_oom()
        assert new_batch_size == 1
        
        # Second OOM should stay at 1 (minimum)
        new_batch_size = batcher.handle_oom()
        assert new_batch_size == 1
    
    def test_step_successful(self):
        """Test successful step recording."""
        batcher = DynamicBatchSizer(16, patience=10)  # High patience to prevent auto-increase
        
        # First successful step
        batch_size = batcher.step_successful()
        assert batch_size == 16
        assert batcher.successful_steps == 1
        
        # Second successful step
        batch_size = batcher.step_successful()
        assert batch_size == 16
        assert batcher.successful_steps == 2
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_step_successful_with_memory_check(self, mock_cuda):
        """Test successful step with memory usage check."""
        batcher = DynamicBatchSizer(16, patience=1, memory_threshold=0.5)
        
        # Mock low memory usage
        with patch.object(batcher.profiler, 'get_memory_stats') as mock_stats:
            mock_stats.return_value = {
                'system': {'percent': 30.0},  # Low memory usage
                'process': {'rss': 1000000}
            }
            
            batch_size = batcher.step_successful()
            
            # Should increase batch size due to low memory usage
            assert batch_size > 16
    
    def test_get_current_batch_size(self):
        """Test getting current batch size."""
        batcher = DynamicBatchSizer(32)
        assert batcher.get_current_batch_size() == 32
    
    def test_reset(self):
        """Test resetting counters."""
        batcher = DynamicBatchSizer(32)
        
        # Simulate some activity
        batcher.handle_oom()
        batcher.step_successful()
        
        assert batcher.oom_count > 0
        assert batcher.successful_steps > 0
        
        # Reset
        batcher.reset()
        
        assert batcher.oom_count == 0
        assert batcher.successful_steps == 0


class TestMemoryEfficientForward:
    """Test memory_efficient_forward decorator."""
    
    def test_memory_efficient_forward_decorator(self):
        """Test that decorator works correctly."""
        
        @memory_efficient_forward
        def dummy_forward(x):
            return x * 2
        
        x = torch.tensor([1, 2, 3])
        result = dummy_forward(x)
        
        assert torch.equal(result, torch.tensor([2, 4, 6]))
    
    def test_memory_efficient_forward_oom_handling(self):
        """Test OOM handling in decorator."""
        
        @memory_efficient_forward
        def oom_forward(x):
            raise RuntimeError("CUDA out of memory")
        
        x = torch.tensor([1, 2, 3])
        
        with pytest.raises(RuntimeError, match="out of memory"):
            oom_forward(x)
    
    def test_memory_efficient_forward_other_errors(self):
        """Test that other errors are not caught."""
        
        @memory_efficient_forward
        def error_forward(x):
            raise ValueError("Some other error")
        
        x = torch.tensor([1, 2, 3])
        
        with pytest.raises(ValueError, match="Some other error"):
            error_forward(x)


class TestMemoryOptimizer:
    """Test MemoryOptimizer functionality."""
    
    def test_initialization(self, training_config):
        """Test MemoryOptimizer initialization."""
        optimizer = MemoryOptimizer(training_config)
        
        assert optimizer.config == training_config
        assert optimizer.profiler is not None
        assert optimizer.dynamic_batcher is not None
        assert optimizer.gradient_checkpointing_enabled == training_config.gradient_checkpointing
        assert optimizer.mixed_precision_enabled == training_config.use_mixed_precision
    
    def test_optimize_model(self, training_config, simple_model):
        """Test model optimization."""
        optimizer = MemoryOptimizer(training_config)
        
        # Add a mock MambaBlock to the model
        simple_model.mamba_block = MockMambaBlock()
        
        optimized_model = optimizer.optimize_model(simple_model)
        
        # Check that gradient checkpointing was enabled
        assert hasattr(simple_model.mamba_block, 'gradient_checkpointing')
        assert simple_model.mamba_block.gradient_checkpointing is True
    
    def test_handle_oom_error(self, training_config):
        """Test OOM error handling."""
        optimizer = MemoryOptimizer(training_config)
        
        error = RuntimeError("CUDA out of memory")
        result = optimizer.handle_oom_error(error)
        
        assert 'new_batch_size' in result
        assert 'oom_count' in result
        assert 'action' in result
        assert result['action'] == 'batch_size_reduced'
        assert result['new_batch_size'] < training_config.batch_size
    
    def test_step_completed(self, training_config):
        """Test step completion recording."""
        optimizer = MemoryOptimizer(training_config)
        
        result = optimizer.step_completed()
        
        assert 'batch_size' in result
        assert 'successful_steps' in result
        assert result['successful_steps'] == 1
    
    def test_clear_memory(self, training_config):
        """Test memory clearing."""
        optimizer = MemoryOptimizer(training_config)
        
        # This should not raise any errors
        optimizer.clear_memory()
    
    def test_get_memory_report(self, training_config):
        """Test memory report generation."""
        optimizer = MemoryOptimizer(training_config)
        
        report = optimizer.get_memory_report()
        
        assert 'memory_stats' in report
        assert 'current_batch_size' in report
        assert 'oom_count' in report
        assert 'successful_steps' in report
        assert 'gradient_checkpointing' in report
        assert 'mixed_precision' in report
        
        assert report['gradient_checkpointing'] == training_config.gradient_checkpointing
        assert report['mixed_precision'] == training_config.use_mixed_precision
    
    def test_profile_step_context_manager(self, training_config):
        """Test step profiling context manager."""
        optimizer = MemoryOptimizer(training_config)
        
        with optimizer.profile_step("test_step"):
            # Allocate some memory
            dummy_tensor = torch.randn(100, 100)
        
        # Check that profiling was recorded
        assert len(optimizer.profiler.memory_history) >= 2
        
        # Clean up
        del dummy_tensor


class TestFactoryFunction:
    """Test factory function."""
    
    def test_create_memory_optimizer(self, training_config):
        """Test factory function."""
        optimizer = create_memory_optimizer(training_config)
        
        assert isinstance(optimizer, MemoryOptimizer)
        assert optimizer.config == training_config


class TestIntegration:
    """Integration tests for memory optimization."""
    
    def test_full_memory_optimization_workflow(self, training_config):
        """Test complete memory optimization workflow."""
        # Use high patience to prevent automatic batch size increases
        training_config.batch_size = 16
        optimizer = MemoryOptimizer(training_config)
        optimizer.dynamic_batcher.patience = 100  # Prevent auto-increase
        
        model = SimpleModel()
        
        # Add mock MambaBlock
        model.mamba_block = MockMambaBlock()
        
        # Optimize model
        optimized_model = optimizer.optimize_model(model)
        
        # Simulate training steps
        for i in range(5):
            with optimizer.profile_step(f"step_{i}"):
                x = torch.randn(training_config.batch_size, 10)
                output = optimized_model(x)
                loss = output.sum()
                loss.backward()
            
            # Record successful step
            step_result = optimizer.step_completed()
            assert 'batch_size' in step_result
        
        # Get final report
        report = optimizer.get_memory_report()
        assert report['successful_steps'] == 5
        assert report['oom_count'] == 0
    
    def test_oom_recovery_workflow(self, training_config):
        """Test OOM recovery workflow."""
        optimizer = MemoryOptimizer(training_config)
        optimizer.dynamic_batcher.patience = 100  # Prevent auto-increase
        
        # Simulate OOM error
        oom_error = RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        recovery_info = optimizer.handle_oom_error(oom_error)
        
        assert recovery_info['new_batch_size'] < training_config.batch_size
        assert recovery_info['oom_count'] == 1
        
        # Simulate successful steps after recovery
        for i in range(3):
            step_result = optimizer.step_completed()
        
        # Check final state
        report = optimizer.get_memory_report()
        assert report['oom_count'] == 1
        assert report['successful_steps'] == 3


if __name__ == "__main__":
    pytest.main([__file__])