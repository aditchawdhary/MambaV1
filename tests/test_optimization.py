"""Tests for optimization and learning rate scheduling."""

import pytest
import torch
import torch.nn as nn
import math
from unittest.mock import Mock, patch

from mamba_training.training.optimization import (
    OptimizationManager,
    LearningRateScheduler,
    create_optimization_manager
)
from mamba_training.config import TrainingConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        return self.linear2(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def training_config():
    """Create training configuration for testing."""
    return TrainingConfig(
        batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=10,
        use_mixed_precision=False
    )


@pytest.fixture
def optimization_manager(simple_model, training_config):
    """Create optimization manager for testing."""
    return OptimizationManager(simple_model, training_config, total_steps=100)


class TestOptimizationManager:
    """Test OptimizationManager functionality."""
    
    def test_initialization(self, simple_model, training_config):
        """Test OptimizationManager initialization."""
        manager = OptimizationManager(simple_model, training_config, total_steps=100)
        
        assert manager.config == training_config
        assert manager.model == simple_model
        assert manager.total_steps == 100
        assert manager.current_step == 0
        assert manager.accumulated_steps == 0
        assert manager.optimizer is not None
        assert manager.scheduler is not None
        assert manager.scaler is None  # Mixed precision disabled
    
    def test_initialization_with_mixed_precision(self, simple_model, training_config):
        """Test initialization with mixed precision enabled."""
        training_config.use_mixed_precision = True
        manager = OptimizationManager(simple_model, training_config, total_steps=100)
        
        # Scaler is only created if CUDA is available
        if torch.cuda.is_available():
            assert manager.scaler is not None
        else:
            assert manager.scaler is None
    
    def test_optimizer_parameter_groups(self, simple_model, training_config):
        """Test that optimizer correctly separates decay and no-decay parameters."""
        manager = OptimizationManager(simple_model, training_config)
        
        # Should have two parameter groups
        assert len(manager.optimizer.param_groups) == 2
        
        # Check weight decay settings
        decay_group = manager.optimizer.param_groups[0]
        no_decay_group = manager.optimizer.param_groups[1]
        
        assert decay_group['weight_decay'] == training_config.weight_decay
        assert no_decay_group['weight_decay'] == 0.0
    
    def test_learning_rate_scheduler_creation(self, simple_model, training_config):
        """Test learning rate scheduler creation."""
        manager = OptimizationManager(simple_model, training_config, total_steps=100)
        
        assert manager.scheduler is not None
        
        # Test scheduler without total_steps
        manager_no_steps = OptimizationManager(simple_model, training_config)
        assert manager_no_steps.scheduler is None
    
    def test_gradient_accumulation(self, optimization_manager, simple_model):
        """Test gradient accumulation functionality."""
        # Create dummy input and target
        x = torch.randn(2, 10)
        target = torch.randn(2, 1)
        
        # First accumulation step
        output = simple_model(x)
        loss = nn.MSELoss()(output, target)
        
        metrics = optimization_manager.step(loss)
        
        assert metrics['optimizer_step'] is False
        assert optimization_manager.accumulated_steps == 1
        assert optimization_manager.current_step == 0
        
        # Second accumulation step (should trigger optimizer step)
        output = simple_model(x)
        loss = nn.MSELoss()(output, target)
        
        metrics = optimization_manager.step(loss)
        
        assert metrics['optimizer_step'] is True
        assert optimization_manager.accumulated_steps == 0
        assert optimization_manager.current_step == 1
        assert 'grad_norm' in metrics
        assert 'learning_rate' in metrics
    
    def test_gradient_clipping(self, optimization_manager, simple_model):
        """Test gradient clipping functionality."""
        # Create input that will produce large gradients
        x = torch.randn(2, 10) * 100
        target = torch.randn(2, 1) * 100
        
        # Accumulate gradients to trigger optimizer step
        for _ in range(optimization_manager.config.gradient_accumulation_steps):
            output = simple_model(x)
            loss = nn.MSELoss()(output, target)
            metrics = optimization_manager.step(loss)
        
        # Check that gradient norm is reported
        assert 'grad_norm' in metrics
        assert isinstance(metrics['grad_norm'], float)
        assert metrics['grad_norm'] >= 0
    
    def test_learning_rate_progression(self, optimization_manager):
        """Test learning rate changes during training."""
        initial_lr = optimization_manager.get_current_lr()
        
        # Simulate training steps
        x = torch.randn(2, 10)
        target = torch.randn(2, 1)
        
        lrs = []
        for step in range(20):
            # Accumulate gradients
            for _ in range(optimization_manager.config.gradient_accumulation_steps):
                output = optimization_manager.model(x)
                loss = nn.MSELoss()(output, target)
                optimization_manager.step(loss)
            
            lrs.append(optimization_manager.get_current_lr())
        
        # During warmup, learning rate should increase
        warmup_lrs = lrs[:optimization_manager.config.warmup_steps]
        if len(warmup_lrs) > 1:
            assert warmup_lrs[-1] > warmup_lrs[0]
    
    def test_state_save_load(self, optimization_manager):
        """Test optimizer state saving and loading."""
        # Take some optimization steps
        x = torch.randn(2, 10)
        target = torch.randn(2, 1)
        
        for _ in range(5):
            for _ in range(optimization_manager.config.gradient_accumulation_steps):
                output = optimization_manager.model(x)
                loss = nn.MSELoss()(output, target)
                optimization_manager.step(loss)
        
        # Save state
        state = optimization_manager.get_optimizer_state()
        
        assert 'optimizer_state_dict' in state
        assert 'current_step' in state
        assert 'accumulated_steps' in state
        assert 'scheduler_state_dict' in state
        
        # Create new manager and load state
        new_manager = OptimizationManager(
            optimization_manager.model,
            optimization_manager.config,
            optimization_manager.total_steps
        )
        
        original_step = optimization_manager.current_step
        original_lr = optimization_manager.get_current_lr()
        
        new_manager.load_optimizer_state(state)
        
        assert new_manager.current_step == original_step
        assert abs(new_manager.get_current_lr() - original_lr) < 1e-6
    
    def test_lr_schedule_preview(self, optimization_manager):
        """Test learning rate schedule preview."""
        schedule = optimization_manager.get_lr_schedule_preview(50)
        
        assert isinstance(schedule, dict)
        assert len(schedule) == 50
        
        # Check that all values are non-negative (can be 0 during warmup)
        for step, lr in schedule.items():
            assert isinstance(step, int)
            assert isinstance(lr, float)
            assert lr >= 0
    
    def test_zero_grad(self, optimization_manager, simple_model):
        """Test gradient clearing."""
        # Create gradients
        x = torch.randn(2, 10)
        target = torch.randn(2, 1)
        output = simple_model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check gradients exist
        has_grad = any(p.grad is not None for p in simple_model.parameters())
        assert has_grad
        
        # Clear gradients
        optimization_manager.zero_grad()
        
        # Check gradients are cleared
        for param in simple_model.parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))


class TestLearningRateScheduler:
    """Test LearningRateScheduler utility functions."""
    
    def test_cosine_with_warmup(self):
        """Test cosine learning rate schedule with warmup."""
        warmup_steps = 10
        total_steps = 100
        base_lr = 1.0
        
        # Test warmup phase
        for step in range(warmup_steps):
            lr = LearningRateScheduler.cosine_with_warmup(
                step, warmup_steps, total_steps, base_lr
            )
            expected_lr = base_lr * step / warmup_steps
            assert abs(lr - expected_lr) < 1e-6
        
        # Test cosine phase
        mid_step = (warmup_steps + total_steps) // 2
        mid_lr = LearningRateScheduler.cosine_with_warmup(
            mid_step, warmup_steps, total_steps, base_lr
        )
        
        end_lr = LearningRateScheduler.cosine_with_warmup(
            total_steps - 1, warmup_steps, total_steps, base_lr
        )
        
        # Learning rate should decrease during cosine phase
        assert mid_lr < base_lr
        assert end_lr < mid_lr
        assert end_lr >= 0
    
    def test_cosine_with_min_lr(self):
        """Test cosine schedule with minimum learning rate."""
        warmup_steps = 10
        total_steps = 100
        base_lr = 1.0
        min_lr_ratio = 0.1
        
        end_lr = LearningRateScheduler.cosine_with_warmup(
            total_steps - 1, warmup_steps, total_steps, base_lr, min_lr_ratio
        )
        
        # Should not go below minimum
        assert end_lr >= base_lr * min_lr_ratio
    
    def test_linear_with_warmup(self):
        """Test linear learning rate schedule with warmup."""
        warmup_steps = 10
        total_steps = 100
        base_lr = 1.0
        
        # Test warmup phase
        for step in range(warmup_steps):
            lr = LearningRateScheduler.linear_with_warmup(
                step, warmup_steps, total_steps, base_lr
            )
            expected_lr = base_lr * step / warmup_steps
            assert abs(lr - expected_lr) < 1e-6
        
        # Test linear decay phase
        mid_step = (warmup_steps + total_steps) // 2
        mid_lr = LearningRateScheduler.linear_with_warmup(
            mid_step, warmup_steps, total_steps, base_lr
        )
        
        end_lr = LearningRateScheduler.linear_with_warmup(
            total_steps - 1, warmup_steps, total_steps, base_lr
        )
        
        # Learning rate should decrease linearly
        assert mid_lr < base_lr
        assert end_lr < mid_lr
        assert end_lr >= 0


class TestFactoryFunction:
    """Test factory function for creating OptimizationManager."""
    
    def test_create_optimization_manager(self, simple_model, training_config):
        """Test factory function."""
        manager = create_optimization_manager(simple_model, training_config, total_steps=100)
        
        assert isinstance(manager, OptimizationManager)
        assert manager.model == simple_model
        assert manager.config == training_config
        assert manager.total_steps == 100


@pytest.mark.parametrize("use_mixed_precision", [True, False])
def test_mixed_precision_handling(simple_model, use_mixed_precision):
    """Test mixed precision training handling."""
    config = TrainingConfig(use_mixed_precision=use_mixed_precision)
    manager = OptimizationManager(simple_model, config)
    
    if use_mixed_precision and torch.cuda.is_available():
        assert manager.scaler is not None
    else:
        assert manager.scaler is None


def test_edge_cases():
    """Test edge cases and error conditions."""
    model = SimpleModel()
    config = TrainingConfig()
    
    # Test with zero total steps - should have no scheduler but still return constant LR
    manager = OptimizationManager(model, config, total_steps=0)
    assert manager.scheduler is None
    schedule = manager.get_lr_schedule_preview(10)
    assert len(schedule) == 10
    assert all(lr == config.learning_rate for lr in schedule.values())
    
    # Test with no scheduler
    manager_no_scheduler = OptimizationManager(model, config)
    schedule = manager_no_scheduler.get_lr_schedule_preview(10)
    assert all(lr == config.learning_rate for lr in schedule.values())


if __name__ == "__main__":
    pytest.main([__file__])