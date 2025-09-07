"""Tests for error handling and recovery mechanisms."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

import torch
import torch.nn as nn
from torch.optim import AdamW

from mamba_training.training.error_handling import (
    ErrorType,
    ErrorEvent,
    GradientMonitor,
    LossMonitor,
    GradientClippingRecovery,
    LearningRateReductionRecovery,
    CheckpointRecovery,
    ModelReinitializationRecovery,
    ErrorHandler,
    create_error_handler
)
from mamba_training.training.checkpoint_manager import CheckpointManager
from mamba_training.training.monitoring import TrainingLogger
from mamba_training.config import Config


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
def model():
    """Create test model."""
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    """Create test optimizer."""
    return AdamW(model.parameters(), lr=1e-4)


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def checkpoint_manager(temp_dir, config):
    """Create checkpoint manager for testing."""
    return CheckpointManager(temp_dir, config)


@pytest.fixture
def logger(temp_dir):
    """Create logger for testing."""
    return TrainingLogger(temp_dir, "test_error_handling", tensorboard_log=False)


class TestErrorEvent:
    """Test ErrorEvent class."""
    
    def test_error_event_initialization(self):
        """Test error event initialization."""
        event = ErrorEvent(
            error_type=ErrorType.GRADIENT_EXPLOSION,
            step=100,
            epoch=1,
            error_message="Test error",
            traceback_str="Test traceback",
            timestamp=1000.0
        )
        
        assert event.error_type == ErrorType.GRADIENT_EXPLOSION
        assert event.step == 100
        assert event.epoch == 1
        assert event.error_message == "Test error"
        assert event.recovery_attempted is False
        assert event.recovery_successful is False
    
    def test_error_event_auto_timestamp(self):
        """Test automatic timestamp setting."""
        import time
        start_time = time.time()
        
        event = ErrorEvent(
            error_type=ErrorType.NAN_LOSS,
            step=50,
            epoch=1,
            error_message="Test",
            traceback_str="Test",
            timestamp=0
        )
        
        end_time = time.time()
        assert start_time <= event.timestamp <= end_time


class TestGradientMonitor:
    """Test GradientMonitor class."""
    
    def test_monitor_initialization(self):
        """Test gradient monitor initialization."""
        monitor = GradientMonitor(
            explosion_threshold=5.0,
            nan_check=True,
            inf_check=True,
            history_size=50
        )
        
        assert monitor.explosion_threshold == 5.0
        assert monitor.nan_check is True
        assert monitor.inf_check is True
        assert monitor.history_size == 50
        assert len(monitor.gradient_norms) == 0
    
    def test_check_gradients_normal(self, model):
        """Test gradient checking with normal gradients."""
        monitor = GradientMonitor(explosion_threshold=100.0)  # Higher threshold for normal gradients
        
        # Create normal gradients
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        error_type = monitor.check_gradients(model)
        assert error_type is None
        assert len(monitor.gradient_norms) == 1
        assert monitor.gradient_norms[0] > 0
    
    def test_check_gradients_explosion(self, model):
        """Test gradient explosion detection."""
        monitor = GradientMonitor(explosion_threshold=1.0)
        
        # Create large gradients manually
        for param in model.parameters():
            param.grad = torch.ones_like(param) * 10.0
        
        error_type = monitor.check_gradients(model)
        assert error_type == ErrorType.GRADIENT_EXPLOSION
    
    def test_check_gradients_nan(self, model):
        """Test NaN gradient detection."""
        monitor = GradientMonitor()
        
        # Create NaN gradients
        for param in model.parameters():
            param.grad = torch.full_like(param, float('nan'))
        
        error_type = monitor.check_gradients(model)
        assert error_type == ErrorType.NAN_LOSS
    
    def test_check_gradients_inf(self, model):
        """Test Inf gradient detection."""
        monitor = GradientMonitor()
        
        # Create Inf gradients
        for param in model.parameters():
            param.grad = torch.full_like(param, float('inf'))
        
        error_type = monitor.check_gradients(model)
        assert error_type == ErrorType.INF_LOSS
    
    def test_check_gradients_no_gradients(self, model):
        """Test gradient checking with no gradients."""
        monitor = GradientMonitor()
        
        # Model without gradients
        error_type = monitor.check_gradients(model)
        assert error_type is None
        assert len(monitor.gradient_norms) == 0
    
    def test_gradient_stats(self, model):
        """Test gradient statistics computation."""
        monitor = GradientMonitor()
        
        # Add some gradient norms
        monitor.gradient_norms = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        stats = monitor.get_gradient_stats()
        
        assert stats['current_norm'] == 5.0
        assert stats['mean_norm'] == 3.0
        assert stats['max_norm'] == 5.0
        assert stats['min_norm'] == 1.0
    
    def test_gradient_stats_empty(self):
        """Test gradient statistics with empty history."""
        monitor = GradientMonitor()
        
        stats = monitor.get_gradient_stats()
        assert stats == {}
    
    def test_history_size_limit(self, model):
        """Test gradient history size limit."""
        monitor = GradientMonitor(history_size=3)
        
        # Add more gradients than history size
        for i in range(5):
            for param in model.parameters():
                param.grad = torch.ones_like(param) * (i + 1)
            monitor.check_gradients(model)
        
        # Should only keep last 3
        assert len(monitor.gradient_norms) == 3
    
    def test_reset(self, model):
        """Test gradient monitor reset."""
        monitor = GradientMonitor()
        
        # Add some data
        monitor.gradient_norms = [1.0, 2.0, 3.0]
        monitor.error_count = 5
        
        monitor.reset()
        
        assert len(monitor.gradient_norms) == 0
        assert monitor.error_count == 0


class TestLossMonitor:
    """Test LossMonitor class."""
    
    def test_monitor_initialization(self):
        """Test loss monitor initialization."""
        monitor = LossMonitor(
            nan_check=True,
            inf_check=True,
            divergence_threshold=50.0,
            history_size=50
        )
        
        assert monitor.nan_check is True
        assert monitor.inf_check is True
        assert monitor.divergence_threshold == 50.0
        assert monitor.history_size == 50
        assert len(monitor.loss_history) == 0
        assert monitor.min_loss == float('inf')
    
    def test_check_loss_normal(self):
        """Test loss checking with normal loss."""
        monitor = LossMonitor()
        
        loss = torch.tensor(0.5)
        error_type = monitor.check_loss(loss)
        
        assert error_type is None
        assert len(monitor.loss_history) == 1
        assert monitor.loss_history[0] == 0.5
        assert monitor.min_loss == 0.5
    
    def test_check_loss_nan(self):
        """Test NaN loss detection."""
        monitor = LossMonitor()
        
        loss = torch.tensor(float('nan'))
        error_type = monitor.check_loss(loss)
        
        assert error_type == ErrorType.NAN_LOSS
    
    def test_check_loss_inf(self):
        """Test Inf loss detection."""
        monitor = LossMonitor()
        
        loss = torch.tensor(float('inf'))
        error_type = monitor.check_loss(loss)
        
        assert error_type == ErrorType.INF_LOSS
    
    def test_check_loss_divergence(self):
        """Test loss divergence detection."""
        monitor = LossMonitor(divergence_threshold=10.0)
        
        # Set minimum loss
        monitor.check_loss(torch.tensor(1.0))
        
        # Check for divergence
        divergent_loss = torch.tensor(15.0)  # 1.0 + 10.0 + margin
        error_type = monitor.check_loss(divergent_loss)
        
        assert error_type == ErrorType.MODEL_DIVERGENCE
    
    def test_loss_stats(self):
        """Test loss statistics computation."""
        monitor = LossMonitor()
        
        # Add some losses
        losses = [1.0, 2.0, 0.5, 3.0, 1.5]
        for loss_val in losses:
            monitor.check_loss(torch.tensor(loss_val))
        
        stats = monitor.get_loss_stats()
        
        assert stats['current_loss'] == 1.5
        assert stats['mean_loss'] == sum(losses) / len(losses)
        assert stats['min_loss'] == 0.5
        assert stats['max_loss'] == 3.0
    
    def test_loss_stats_empty(self):
        """Test loss statistics with empty history."""
        monitor = LossMonitor()
        
        stats = monitor.get_loss_stats()
        assert stats == {}
    
    def test_history_size_limit(self):
        """Test loss history size limit."""
        monitor = LossMonitor(history_size=3)
        
        # Add more losses than history size
        for i in range(5):
            monitor.check_loss(torch.tensor(float(i)))
        
        # Should only keep last 3
        assert len(monitor.loss_history) == 3
        assert monitor.loss_history == [2.0, 3.0, 4.0]
    
    def test_reset(self):
        """Test loss monitor reset."""
        monitor = LossMonitor()
        
        # Add some data
        monitor.loss_history = [1.0, 2.0, 3.0]
        monitor.min_loss = 1.0
        
        monitor.reset()
        
        assert len(monitor.loss_history) == 0
        assert monitor.min_loss == float('inf')


class TestRecoveryStrategies:
    """Test recovery strategy classes."""
    
    def test_gradient_clipping_recovery(self, model, optimizer):
        """Test gradient clipping recovery strategy."""
        strategy = GradientClippingRecovery(max_norm=1.0)
        
        # Create error event
        error_event = ErrorEvent(
            error_type=ErrorType.GRADIENT_EXPLOSION,
            step=100,
            epoch=1,
            error_message="Test",
            traceback_str="Test",
            timestamp=1000.0
        )
        
        # Test can_recover
        assert strategy.can_recover(error_event) is True
        
        # Create large gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param) * 10.0
        
        # Test recovery
        success = strategy.recover(model, optimizer, None, error_event)
        assert success is True
    
    def test_learning_rate_reduction_recovery(self, model, optimizer):
        """Test learning rate reduction recovery strategy."""
        strategy = LearningRateReductionRecovery(reduction_factor=0.5, min_lr=1e-8)
        
        # Create error event
        error_event = ErrorEvent(
            error_type=ErrorType.GRADIENT_EXPLOSION,
            step=100,
            epoch=1,
            error_message="Test",
            traceback_str="Test",
            timestamp=1000.0
        )
        
        # Test can_recover
        assert strategy.can_recover(error_event) is True
        
        # Get initial learning rate
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Test recovery
        success = strategy.recover(model, optimizer, None, error_event)
        assert success is True
        
        # Check learning rate was reduced
        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr == initial_lr * 0.5
    
    def test_learning_rate_min_limit(self, model, optimizer):
        """Test learning rate reduction minimum limit."""
        strategy = LearningRateReductionRecovery(reduction_factor=0.1, min_lr=1e-6)
        
        # Set very low learning rate
        optimizer.param_groups[0]['lr'] = 1e-7
        
        error_event = ErrorEvent(
            error_type=ErrorType.GRADIENT_EXPLOSION,
            step=100,
            epoch=1,
            error_message="Test",
            traceback_str="Test",
            timestamp=1000.0
        )
        
        # Test recovery
        success = strategy.recover(model, optimizer, None, error_event)
        assert success is True
        
        # Learning rate should be clamped to minimum
        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr == 1e-6
    
    def test_checkpoint_recovery(self, model, optimizer, checkpoint_manager):
        """Test checkpoint recovery strategy."""
        strategy = CheckpointRecovery(checkpoint_manager)
        
        # Save a checkpoint first
        checkpoint_manager.save_checkpoint(
            model, optimizer, None, 1, 100, 0.5
        )
        
        # Create error event
        error_event = ErrorEvent(
            error_type=ErrorType.MODEL_DIVERGENCE,
            step=200,
            epoch=2,
            error_message="Test",
            traceback_str="Test",
            timestamp=1000.0
        )
        
        # Test can_recover
        assert strategy.can_recover(error_event) is True
        
        # Test recovery
        success = strategy.recover(model, optimizer, None, error_event)
        assert success is True
    
    def test_checkpoint_recovery_no_checkpoint(self, model, optimizer, checkpoint_manager):
        """Test checkpoint recovery with no available checkpoint."""
        strategy = CheckpointRecovery(checkpoint_manager)
        
        error_event = ErrorEvent(
            error_type=ErrorType.MODEL_DIVERGENCE,
            step=100,
            epoch=1,
            error_message="Test",
            traceback_str="Test",
            timestamp=1000.0
        )
        
        # Test recovery without checkpoint
        success = strategy.recover(model, optimizer, None, error_event)
        assert success is False
    
    def test_model_reinitialization_recovery(self, model, optimizer):
        """Test model reinitialization recovery strategy."""
        strategy = ModelReinitializationRecovery()
        
        # Store initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone()
        
        # Create error event
        error_event = ErrorEvent(
            error_type=ErrorType.NAN_LOSS,
            step=100,
            epoch=1,
            error_message="Test",
            traceback_str="Test",
            timestamp=1000.0
        )
        
        # Test can_recover
        assert strategy.can_recover(error_event) is True
        
        # Test recovery
        success = strategy.recover(model, optimizer, None, error_event)
        assert success is True
        
        # Check that parameters changed (reinitialized)
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                params_changed = True
                break
        
        assert params_changed is True
    
    def test_model_reinitialization_specific_layers(self, model, optimizer):
        """Test model reinitialization with specific layers."""
        strategy = ModelReinitializationRecovery(reinit_layers=['linear'])
        
        error_event = ErrorEvent(
            error_type=ErrorType.NAN_LOSS,
            step=100,
            epoch=1,
            error_message="Test",
            traceback_str="Test",
            timestamp=1000.0
        )
        
        # Test recovery
        success = strategy.recover(model, optimizer, None, error_event)
        assert success is True


class TestErrorHandler:
    """Test ErrorHandler class."""
    
    def test_error_handler_initialization(self, checkpoint_manager, logger):
        """Test error handler initialization."""
        handler = ErrorHandler(
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            max_recovery_attempts=5
        )
        
        assert handler.checkpoint_manager == checkpoint_manager
        assert handler.logger == logger
        assert handler.max_recovery_attempts == 5
        assert len(handler.recovery_strategies) > 0
        assert len(handler.error_history) == 0
        assert handler.recovery_attempts == 0
    
    def test_check_for_errors_normal(self, model):
        """Test error checking with normal conditions."""
        handler = ErrorHandler()
        handler.gradient_monitor.explosion_threshold = 100.0  # Higher threshold for normal gradients
        
        # Create normal gradients and loss
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        error_event = handler.check_for_errors(model, loss, step=100, epoch=1)
        assert error_event is None
    
    def test_check_for_errors_gradient_explosion(self, model):
        """Test error detection for gradient explosion."""
        handler = ErrorHandler()
        handler.gradient_monitor.explosion_threshold = 1.0
        
        # Create large gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param) * 10.0
        
        loss = torch.tensor(0.5)
        error_event = handler.check_for_errors(model, loss, step=100, epoch=1)
        
        assert error_event is not None
        assert error_event.error_type == ErrorType.GRADIENT_EXPLOSION
        assert error_event.step == 100
        assert error_event.epoch == 1
    
    def test_check_for_errors_nan_loss(self, model):
        """Test error detection for NaN loss."""
        handler = ErrorHandler()
        
        loss = torch.tensor(float('nan'))
        error_event = handler.check_for_errors(model, loss, step=100, epoch=1)
        
        assert error_event is not None
        assert error_event.error_type == ErrorType.NAN_LOSS
    
    def test_handle_error_successful_recovery(self, model, optimizer):
        """Test successful error handling and recovery."""
        handler = ErrorHandler()
        
        # Create error event
        error_event = ErrorEvent(
            error_type=ErrorType.GRADIENT_EXPLOSION,
            step=100,
            epoch=1,
            error_message="Test error",
            traceback_str="Test traceback",
            timestamp=1000.0
        )
        
        # Test error handling
        success = handler.handle_error(error_event, model, optimizer)
        
        assert success is True
        assert error_event.recovery_attempted is True
        assert error_event.recovery_successful is True
        assert error_event.recovery_method is not None
        assert len(handler.error_history) == 1
        assert handler.recovery_attempts == 1
    
    def test_handle_error_max_attempts_exceeded(self, model, optimizer):
        """Test error handling when max attempts exceeded."""
        handler = ErrorHandler(max_recovery_attempts=1)
        
        # Exceed max attempts
        handler.recovery_attempts = 2
        
        error_event = ErrorEvent(
            error_type=ErrorType.GRADIENT_EXPLOSION,
            step=100,
            epoch=1,
            error_message="Test error",
            traceback_str="Test traceback",
            timestamp=1000.0
        )
        
        success = handler.handle_error(error_event, model, optimizer)
        assert success is False
    
    def test_handle_error_no_suitable_strategy(self, model, optimizer):
        """Test error handling with no suitable recovery strategy."""
        # Create handler with no strategies
        handler = ErrorHandler(recovery_strategies=[])
        
        error_event = ErrorEvent(
            error_type=ErrorType.GRADIENT_EXPLOSION,
            step=100,
            epoch=1,
            error_message="Test error",
            traceback_str="Test traceback",
            timestamp=1000.0
        )
        
        success = handler.handle_error(error_event, model, optimizer)
        
        assert success is False
        assert error_event.recovery_attempted is True
        assert error_event.recovery_successful is False
    
    def test_get_error_statistics(self, model, optimizer):
        """Test error statistics computation."""
        handler = ErrorHandler()
        
        # Add some error events
        for i in range(3):
            error_event = ErrorEvent(
                error_type=ErrorType.GRADIENT_EXPLOSION,
                step=100 + i,
                epoch=1,
                error_message=f"Test error {i}",
                traceback_str="Test traceback",
                timestamp=1000.0 + i
            )
            handler.handle_error(error_event, model, optimizer)
        
        stats = handler.get_error_statistics()
        
        assert stats['total_errors'] == 3
        assert stats['error_counts']['gradient_explosion'] == 3
        assert stats['recovery_success_rate'] == 1.0  # All should succeed
        assert stats['recovery_attempts'] == 3
    
    def test_get_error_statistics_empty(self):
        """Test error statistics with no errors."""
        handler = ErrorHandler()
        
        stats = handler.get_error_statistics()
        assert stats == {}
    
    def test_reset(self, model, optimizer):
        """Test error handler reset."""
        handler = ErrorHandler()
        
        # Add some data
        error_event = ErrorEvent(
            error_type=ErrorType.GRADIENT_EXPLOSION,
            step=100,
            epoch=1,
            error_message="Test error",
            traceback_str="Test traceback",
            timestamp=1000.0
        )
        handler.handle_error(error_event, model, optimizer)
        
        # Reset
        handler.reset()
        
        assert len(handler.error_history) == 0
        assert handler.recovery_attempts == 0
        assert len(handler.gradient_monitor.gradient_norms) == 0
        assert len(handler.loss_monitor.loss_history) == 0


class TestCreateErrorHandler:
    """Test create_error_handler function."""
    
    def test_create_error_handler_basic(self):
        """Test creating error handler with basic configuration."""
        handler = create_error_handler()
        
        assert isinstance(handler, ErrorHandler)
        assert handler.checkpoint_manager is None
        assert handler.logger is None
        assert handler.max_recovery_attempts == 3
        assert len(handler.recovery_strategies) >= 2  # At least gradient clipping and LR reduction
    
    def test_create_error_handler_with_checkpoint_manager(self, checkpoint_manager):
        """Test creating error handler with checkpoint manager."""
        handler = create_error_handler(checkpoint_manager=checkpoint_manager)
        
        assert handler.checkpoint_manager == checkpoint_manager
        # Should have checkpoint recovery strategy
        strategy_names = [s.name for s in handler.recovery_strategies]
        assert "checkpoint_recovery" in strategy_names
    
    def test_create_error_handler_with_logger(self, logger):
        """Test creating error handler with logger."""
        handler = create_error_handler(logger=logger)
        
        assert handler.logger == logger
    
    def test_create_error_handler_custom_parameters(self):
        """Test creating error handler with custom parameters."""
        handler = create_error_handler(
            gradient_clip_norm=2.0,
            lr_reduction_factor=0.3,
            max_recovery_attempts=5
        )
        
        assert handler.max_recovery_attempts == 5
        
        # Check gradient clipping strategy parameters
        gradient_strategy = None
        for strategy in handler.recovery_strategies:
            if strategy.name == "gradient_clipping":
                gradient_strategy = strategy
                break
        
        assert gradient_strategy is not None
        assert gradient_strategy.max_norm == 2.0
        
        # Check learning rate reduction strategy parameters
        lr_strategy = None
        for strategy in handler.recovery_strategies:
            if strategy.name == "learning_rate_reduction":
                lr_strategy = strategy
                break
        
        assert lr_strategy is not None
        assert lr_strategy.reduction_factor == 0.3


class TestIntegration:
    """Integration tests for error handling system."""
    
    def test_end_to_end_error_handling(self, model, optimizer, checkpoint_manager, logger):
        """Test end-to-end error handling workflow."""
        handler = ErrorHandler(
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            max_recovery_attempts=3
        )
        
        # Save initial checkpoint
        checkpoint_manager.save_checkpoint(
            model, optimizer, None, 1, 50, 0.8
        )
        
        # Simulate training step with gradient explosion
        for param in model.parameters():
            param.grad = torch.ones_like(param) * 20.0  # Large gradients
        
        loss = torch.tensor(0.5)
        
        # Check for errors
        error_event = handler.check_for_errors(model, loss, step=100, epoch=2)
        assert error_event is not None
        assert error_event.error_type == ErrorType.GRADIENT_EXPLOSION
        
        # Handle error
        success = handler.handle_error(error_event, model, optimizer)
        assert success is True
        
        # Verify recovery
        assert error_event.recovery_successful is True
        assert len(handler.error_history) == 1
        
        # Get statistics
        stats = handler.get_error_statistics()
        assert stats['total_errors'] == 1
        assert stats['recovery_success_rate'] == 1.0
    
    def test_multiple_error_types(self, model, optimizer):
        """Test handling multiple types of errors."""
        handler = ErrorHandler()
        
        # Test gradient explosion
        for param in model.parameters():
            param.grad = torch.ones_like(param) * 20.0
        
        loss = torch.tensor(0.5)
        error_event = handler.check_for_errors(model, loss, step=100, epoch=1)
        assert error_event.error_type == ErrorType.GRADIENT_EXPLOSION
        
        success = handler.handle_error(error_event, model, optimizer)
        assert success is True
        
        # Test NaN loss
        nan_loss = torch.tensor(float('nan'))
        error_event = handler.check_for_errors(model, nan_loss, step=101, epoch=1)
        assert error_event.error_type == ErrorType.NAN_LOSS
        
        success = handler.handle_error(error_event, model, optimizer)
        assert success is True
        
        # Check statistics
        stats = handler.get_error_statistics()
        assert stats['total_errors'] == 2
        assert 'gradient_explosion' in stats['error_counts']
        assert 'nan_loss' in stats['error_counts']
    
    def test_recovery_strategy_fallback(self, model, optimizer):
        """Test fallback between recovery strategies."""
        # Create a mock strategy that always fails
        class FailingStrategy:
            def __init__(self):
                self.name = "failing_strategy"
            
            def can_recover(self, error_event):
                return True
            
            def recover(self, model, optimizer, scheduler, error_event, **kwargs):
                raise Exception("Recovery failed")
        
        # Create handler with failing strategy first, then working strategy
        failing_strategy = FailingStrategy()
        working_strategy = GradientClippingRecovery()
        
        handler = ErrorHandler(
            recovery_strategies=[failing_strategy, working_strategy]
        )
        
        error_event = ErrorEvent(
            error_type=ErrorType.GRADIENT_EXPLOSION,
            step=100,
            epoch=1,
            error_message="Test error",
            traceback_str="Test traceback",
            timestamp=1000.0
        )
        
        # Should succeed with second strategy
        success = handler.handle_error(error_event, model, optimizer)
        assert success is True
        assert error_event.recovery_method == "gradient_clipping"


if __name__ == "__main__":
    pytest.main([__file__])