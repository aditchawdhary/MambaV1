"""Tests for training monitoring and logging functionality."""

import json
import tempfile
import shutil
import time
import math
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

import torch
import torch.nn as nn

from mamba_training.training.monitoring import (
    MetricValue,
    TrainingMetrics,
    MetricsTracker,
    TrainingLogger,
    ProgressReporter,
    compute_perplexity,
    compute_gradient_norm,
    create_training_monitor
)
from mamba_training.config import Config, MambaConfig, TrainingConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        model=MambaConfig(d_model=128, n_layers=2),
        training=TrainingConfig(batch_size=16, learning_rate=1e-4),
        experiment_name="test_experiment",
        log_level="INFO"
    )


@pytest.fixture
def model():
    """Create test model."""
    return SimpleModel()


class TestMetricValue:
    """Test MetricValue class."""
    
    def test_metric_value_initialization(self):
        """Test metric value initialization."""
        metric = MetricValue(value=0.5, timestamp=1000.0, step=100, epoch=1)
        
        assert metric.value == 0.5
        assert metric.timestamp == 1000.0
        assert metric.step == 100
        assert metric.epoch == 1
    
    def test_metric_value_auto_timestamp(self):
        """Test automatic timestamp setting."""
        start_time = time.time()
        metric = MetricValue(value=0.5, timestamp=0, step=100)
        end_time = time.time()
        
        assert start_time <= metric.timestamp <= end_time


class TestTrainingMetrics:
    """Test TrainingMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization with defaults."""
        metrics = TrainingMetrics()
        
        assert metrics.loss == 0.0
        assert metrics.perplexity == 0.0
        assert metrics.throughput_tokens_per_sec == 0.0
        assert metrics.learning_rate == 0.0
        assert metrics.custom_metrics == {}
    
    def test_metrics_with_custom_values(self):
        """Test metrics with custom values."""
        custom_metrics = {'accuracy': 0.95, 'f1_score': 0.88}
        metrics = TrainingMetrics(
            loss=0.5,
            perplexity=1.65,
            learning_rate=1e-4,
            custom_metrics=custom_metrics
        )
        
        assert metrics.loss == 0.5
        assert metrics.perplexity == 1.65
        assert metrics.learning_rate == 1e-4
        assert metrics.custom_metrics == custom_metrics
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        custom_metrics = {'accuracy': 0.95}
        metrics = TrainingMetrics(
            loss=0.5,
            perplexity=1.65,
            custom_metrics=custom_metrics
        )
        
        result = metrics.to_dict()
        
        assert result['loss'] == 0.5
        assert result['perplexity'] == 1.65
        assert result['accuracy'] == 0.95  # Custom metrics should be included


class TestMetricsTracker:
    """Test MetricsTracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = MetricsTracker(window_size=50)
        
        assert tracker.window_size == 50
        assert tracker.global_step == 0
        assert tracker.epoch == 0
        assert len(tracker.metrics_history) == 0
    
    def test_add_metric(self):
        """Test adding metrics."""
        tracker = MetricsTracker()
        
        tracker.add_metric('loss', 0.5)
        tracker.add_metric('accuracy', 0.8)
        
        assert len(tracker.metrics_history['loss']) == 1
        assert len(tracker.metrics_history['accuracy']) == 1
        assert tracker.metrics_history['loss'][0].value == 0.5
        assert tracker.metrics_history['accuracy'][0].value == 0.8
    
    def test_update_loss_and_perplexity(self):
        """Test loss update and perplexity computation."""
        tracker = MetricsTracker()
        
        tracker.update_loss(0.5)
        
        assert len(tracker.metrics_history['loss']) == 1
        assert len(tracker.metrics_history['perplexity']) == 1
        assert tracker.metrics_history['loss'][0].value == 0.5
        
        # Check perplexity calculation
        expected_perplexity = math.exp(0.5)
        assert abs(tracker.metrics_history['perplexity'][0].value - expected_perplexity) < 1e-6
    
    def test_update_loss_overflow_protection(self):
        """Test perplexity overflow protection."""
        tracker = MetricsTracker()
        
        # Very high loss should cap perplexity
        tracker.update_loss(100.0)
        
        perplexity = tracker.metrics_history['perplexity'][0].value
        assert perplexity == 1e6  # Should be capped
    
    def test_timing_methods(self):
        """Test timing measurement methods."""
        tracker = MetricsTracker()
        
        # Test batch timing
        tracker.start_batch()
        time.sleep(0.01)  # Small delay
        tracker.end_batch(batch_size=32, sequence_length=128)
        
        assert len(tracker.metrics_history['batch_time']) == 1
        assert len(tracker.metrics_history['throughput_samples_per_sec']) == 1
        assert len(tracker.metrics_history['throughput_tokens_per_sec']) == 1
        
        batch_time = tracker.metrics_history['batch_time'][0].value
        assert batch_time > 0.005  # Should be at least 5ms
        
        # Test forward/backward timing
        tracker.start_forward()
        time.sleep(0.005)
        tracker.end_forward()
        
        tracker.start_backward()
        time.sleep(0.005)
        tracker.end_backward()
        
        assert len(tracker.metrics_history['forward_time']) == 1
        assert len(tracker.metrics_history['backward_time']) == 1
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        tracker = MetricsTracker()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=1024**3), \
             patch('torch.cuda.memory_reserved', return_value=2*1024**3):
            
            tracker.update_memory_usage()
            
            assert len(tracker.metrics_history['memory_used_gb']) == 1
            assert len(tracker.metrics_history['memory_reserved_gb']) == 1
            assert tracker.metrics_history['memory_used_gb'][0].value == 1.0
            assert tracker.metrics_history['memory_reserved_gb'][0].value == 2.0
    
    def test_get_current_metrics(self):
        """Test getting current metrics."""
        tracker = MetricsTracker()
        
        tracker.add_metric('loss', 0.5)
        tracker.add_metric('accuracy', 0.8)
        tracker.update_learning_rate(1e-4)
        
        metrics = tracker.get_current_metrics()
        
        assert metrics.loss == 0.5
        assert metrics.learning_rate == 1e-4
        assert metrics.custom_metrics['accuracy'] == 0.8
    
    def test_get_averaged_metrics(self):
        """Test getting averaged metrics."""
        tracker = MetricsTracker(window_size=5)
        
        # Add multiple values
        for i in range(10):
            tracker.add_metric('loss', i * 0.1)
        
        # Get averaged metrics over last 3 values
        metrics = tracker.get_averaged_metrics(window_size=3)
        
        # Should average last 3 values: 0.7, 0.8, 0.9
        expected_avg = (0.7 + 0.8 + 0.9) / 3
        assert abs(metrics.loss - expected_avg) < 1e-6
    
    def test_window_size_limit(self):
        """Test that metrics history respects window size."""
        tracker = MetricsTracker(window_size=3)
        
        # Add more values than window size
        for i in range(5):
            tracker.add_metric('loss', i)
        
        # Should only keep last 3 values
        assert len(tracker.metrics_history['loss']) == 3
        values = [mv.value for mv in tracker.metrics_history['loss']]
        assert values == [2, 3, 4]
    
    def test_step_and_epoch_tracking(self):
        """Test step and epoch tracking."""
        tracker = MetricsTracker()
        
        tracker.set_epoch(5)
        tracker.step()
        tracker.step()
        
        assert tracker.epoch == 5
        assert tracker.global_step == 2
    
    def test_reset(self):
        """Test resetting tracker."""
        tracker = MetricsTracker()
        
        tracker.add_metric('loss', 0.5)
        tracker.step()
        tracker.set_epoch(1)
        
        tracker.reset()
        
        assert len(tracker.metrics_history) == 0
        assert tracker.global_step == 0
        assert tracker.epoch == 0


class TestTrainingLogger:
    """Test TrainingLogger class."""
    
    def test_logger_initialization(self, temp_dir):
        """Test logger initialization."""
        logger = TrainingLogger(
            log_dir=temp_dir,
            experiment_name="test_exp",
            log_level="INFO"
        )
        
        assert logger.log_dir == temp_dir
        assert logger.experiment_name == "test_exp"
        assert logger.logger.level == 20  # INFO level
        assert temp_dir.exists()
    
    def test_log_metrics(self, temp_dir):
        """Test metrics logging."""
        logger = TrainingLogger(
            log_dir=temp_dir,
            experiment_name="test_exp",
            tensorboard_log=False  # Disable for testing
        )
        
        metrics = TrainingMetrics(
            loss=0.5,
            perplexity=1.65,
            learning_rate=1e-4,
            custom_metrics={'accuracy': 0.8}
        )
        
        logger.log_metrics(metrics, step=100, epoch=1)
        
        # Check that metrics file was created
        metrics_file = temp_dir / "test_exp_metrics.jsonl"
        assert metrics_file.exists()
        
        # Check metrics file content
        with open(metrics_file, 'r') as f:
            log_entry = json.loads(f.readline())
        
        assert log_entry['step'] == 100
        assert log_entry['epoch'] == 1
        assert log_entry['metrics']['loss'] == 0.5
        assert log_entry['metrics']['accuracy'] == 0.8
    
    def test_log_hyperparameters(self, temp_dir, config):
        """Test hyperparameter logging."""
        logger = TrainingLogger(
            log_dir=temp_dir,
            experiment_name="test_exp",
            tensorboard_log=False
        )
        
        # Should not raise any exceptions
        logger.log_hyperparameters(config)
    
    def test_log_model_info(self, temp_dir, model):
        """Test model information logging."""
        logger = TrainingLogger(
            log_dir=temp_dir,
            experiment_name="test_exp",
            tensorboard_log=False
        )
        
        # Should not raise any exceptions
        logger.log_model_info(model)
    
    def test_log_training_lifecycle(self, temp_dir):
        """Test training lifecycle logging."""
        logger = TrainingLogger(
            log_dir=temp_dir,
            experiment_name="test_exp",
            tensorboard_log=False
        )
        
        metrics = TrainingMetrics(loss=0.3, perplexity=1.35)
        
        # Should not raise any exceptions
        logger.log_training_start(total_steps=1000, total_epochs=5)
        logger.log_training_end(total_time=3600.0, final_metrics=metrics)
    
    def test_log_checkpoint_saved(self, temp_dir):
        """Test checkpoint logging."""
        logger = TrainingLogger(
            log_dir=temp_dir,
            experiment_name="test_exp",
            tensorboard_log=False
        )
        
        checkpoint_path = temp_dir / "checkpoint.pt"
        metrics = TrainingMetrics(loss=0.4)
        
        # Should not raise any exceptions
        logger.log_checkpoint_saved(checkpoint_path, metrics)
    
    def test_log_error(self, temp_dir):
        """Test error logging."""
        logger = TrainingLogger(
            log_dir=temp_dir,
            experiment_name="test_exp",
            tensorboard_log=False
        )
        
        error = ValueError("Test error")
        
        # Should not raise any exceptions
        logger.log_error(error, context="test context")
    
    def test_close(self, temp_dir):
        """Test logger cleanup."""
        logger = TrainingLogger(
            log_dir=temp_dir,
            experiment_name="test_exp"
        )
        
        # Should not raise any exceptions
        logger.close()


class TestProgressReporter:
    """Test ProgressReporter class."""
    
    def test_reporter_initialization(self):
        """Test progress reporter initialization."""
        reporter = ProgressReporter(total_steps=1000, report_interval=100)
        
        assert reporter.total_steps == 1000
        assert reporter.report_interval == 100
        assert reporter.logger is None
    
    def test_report_progress(self):
        """Test progress reporting."""
        mock_logger = Mock()
        mock_logger.logger = Mock()
        
        reporter = ProgressReporter(
            total_steps=1000,
            report_interval=100,
            logger=mock_logger
        )
        
        metrics = TrainingMetrics(
            loss=0.5,
            perplexity=1.65,
            learning_rate=1e-4,
            throughput_tokens_per_sec=1000.0
        )
        
        # Should report at interval
        reporter.report_progress(step=100, metrics=metrics, epoch=1)
        mock_logger.logger.info.assert_called_once()
        
        # Should not report between intervals
        mock_logger.logger.reset_mock()
        reporter.report_progress(step=150, metrics=metrics)
        mock_logger.logger.info.assert_not_called()
    
    def test_create_progress_bar(self):
        """Test progress bar creation."""
        reporter = ProgressReporter(total_steps=100)
        
        # Test 50% progress
        bar = reporter.create_progress_bar(step=50, width=10)
        
        assert "50.0%" in bar
        assert "█" in bar
        assert "░" in bar
    
    def test_progress_without_logger(self, capsys):
        """Test progress reporting without logger."""
        reporter = ProgressReporter(total_steps=100, report_interval=50)
        
        metrics = TrainingMetrics(loss=0.5, perplexity=1.65)
        reporter.report_progress(step=50, metrics=metrics)
        
        captured = capsys.readouterr()
        assert "Step 50/100" in captured.out
        assert "Loss: 0.5000" in captured.out


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_compute_perplexity(self):
        """Test perplexity computation."""
        # Normal case
        ppl = compute_perplexity(0.5)
        expected = math.exp(0.5)
        assert abs(ppl - expected) < 1e-6
        
        # Overflow case
        ppl = compute_perplexity(100.0)
        assert ppl == 1e6  # Should be capped
        
        # Edge case
        ppl = compute_perplexity(0.0)
        assert ppl == 1.0
    
    def test_compute_gradient_norm(self, model):
        """Test gradient norm computation."""
        # Create dummy input and compute gradients
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        grad_norm = compute_gradient_norm(model)
        
        assert grad_norm > 0
        assert isinstance(grad_norm, float)
    
    def test_compute_gradient_norm_no_gradients(self, model):
        """Test gradient norm with no gradients."""
        # Model without gradients
        grad_norm = compute_gradient_norm(model)
        assert grad_norm == 0.0


class TestCreateTrainingMonitor:
    """Test create_training_monitor function."""
    
    def test_create_training_monitor(self, temp_dir, config):
        """Test creating complete monitoring setup."""
        total_steps = 1000
        
        tracker, logger, reporter = create_training_monitor(
            log_dir=temp_dir,
            config=config,
            total_steps=total_steps,
            experiment_name="test_monitor"
        )
        
        assert isinstance(tracker, MetricsTracker)
        assert isinstance(logger, TrainingLogger)
        assert isinstance(reporter, ProgressReporter)
        
        assert tracker.window_size == 100
        assert logger.experiment_name == "test_monitor"
        assert reporter.total_steps == total_steps
    
    def test_create_training_monitor_default_name(self, temp_dir, config):
        """Test creating monitor with default experiment name."""
        tracker, logger, reporter = create_training_monitor(
            log_dir=temp_dir,
            config=config,
            total_steps=1000
        )
        
        assert logger.experiment_name == config.experiment_name


class TestIntegration:
    """Integration tests for monitoring components."""
    
    def test_end_to_end_monitoring(self, temp_dir, config, model):
        """Test end-to-end monitoring workflow."""
        total_steps = 100
        
        # Create monitoring setup
        tracker, logger, reporter = create_training_monitor(
            log_dir=temp_dir,
            config=config,
            total_steps=total_steps
        )
        
        # Log hyperparameters and model info
        logger.log_hyperparameters(config)
        logger.log_model_info(model)
        logger.log_training_start(total_steps, 5)
        
        # Simulate training steps
        for step in range(1, 11):
            tracker.start_batch()
            
            # Simulate forward pass
            tracker.start_forward()
            time.sleep(0.001)
            tracker.end_forward()
            
            # Simulate backward pass
            tracker.start_backward()
            time.sleep(0.001)
            tracker.end_backward()
            
            # Update metrics
            loss = 1.0 - step * 0.05  # Decreasing loss
            tracker.update_loss(loss)
            tracker.update_learning_rate(1e-4)
            tracker.update_gradient_norm(0.5)
            
            tracker.end_batch(batch_size=32, sequence_length=128)
            tracker.step()
            
            # Log metrics
            if step % 5 == 0:
                metrics = tracker.get_current_metrics()
                logger.log_metrics(metrics, step=step, epoch=1)
                reporter.report_progress(step, metrics, epoch=1)
        
        # Finish training
        final_metrics = tracker.get_current_metrics()
        logger.log_training_end(10.0, final_metrics)
        logger.close()
        
        # Verify files were created
        assert (temp_dir / f"{config.experiment_name}.log").exists()
        assert (temp_dir / f"{config.experiment_name}_metrics.jsonl").exists()
    
    def test_metrics_accuracy(self, temp_dir, config):
        """Test accuracy of metric computations."""
        tracker = MetricsTracker()
        
        # Test throughput calculation
        batch_size = 32
        sequence_length = 128
        
        tracker.start_batch()
        start_time = time.time()
        time.sleep(0.1)  # 100ms
        tracker.end_batch(batch_size, sequence_length)
        
        metrics = tracker.get_current_metrics()
        
        # Check throughput calculations
        expected_samples_per_sec = batch_size / 0.1
        expected_tokens_per_sec = (batch_size * sequence_length) / 0.1
        
        # Allow some tolerance for timing variations (10% tolerance)
        samples_tolerance = expected_samples_per_sec * 0.1
        tokens_tolerance = expected_tokens_per_sec * 0.1
        
        assert abs(metrics.throughput_samples_per_sec - expected_samples_per_sec) < samples_tolerance
        assert abs(metrics.throughput_tokens_per_sec - expected_tokens_per_sec) < tokens_tolerance


if __name__ == "__main__":
    pytest.main([__file__])