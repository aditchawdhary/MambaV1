"""Error handling and recovery mechanisms for Mamba training pipeline."""

import logging
import traceback
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .checkpoint_manager import CheckpointManager
from .monitoring import TrainingLogger, MetricsTracker


class ErrorType(Enum):
    """Types of training errors."""
    GRADIENT_EXPLOSION = "gradient_explosion"
    NAN_LOSS = "nan_loss"
    INF_LOSS = "inf_loss"
    OOM_ERROR = "out_of_memory"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    MODEL_DIVERGENCE = "model_divergence"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorEvent:
    """Container for error event information."""
    
    error_type: ErrorType
    step: int
    epoch: int
    error_message: str
    traceback_str: str
    timestamp: float
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp == 0:
            self.timestamp = time.time()


class GradientMonitor:
    """Monitors gradients for explosion and other anomalies."""
    
    def __init__(
        self,
        explosion_threshold: float = 100.0,
        nan_check: bool = True,
        inf_check: bool = True,
        history_size: int = 100
    ):
        """Initialize gradient monitor.
        
        Args:
            explosion_threshold: Threshold for gradient explosion detection
            nan_check: Whether to check for NaN gradients
            inf_check: Whether to check for Inf gradients
            history_size: Size of gradient norm history
        """
        self.explosion_threshold = explosion_threshold
        self.nan_check = nan_check
        self.inf_check = inf_check
        self.history_size = history_size
        
        self.gradient_norms = []
        self.error_count = 0
        
    def check_gradients(self, model: nn.Module) -> Optional[ErrorType]:
        """Check model gradients for anomalies.
        
        Args:
            model: Model to check gradients for
            
        Returns:
            Optional[ErrorType]: Error type if anomaly detected, None otherwise
        """
        total_norm = 0.0
        has_gradients = False
        
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                param_norm = param.grad.data.norm(2)
                
                # Check for NaN
                if self.nan_check and torch.isnan(param_norm):
                    return ErrorType.NAN_LOSS
                
                # Check for Inf
                if self.inf_check and torch.isinf(param_norm):
                    return ErrorType.INF_LOSS
                
                total_norm += param_norm.item() ** 2
        
        if not has_gradients:
            return None
        
        total_norm = total_norm ** 0.5
        
        # Update gradient norm history
        self.gradient_norms.append(total_norm)
        if len(self.gradient_norms) > self.history_size:
            self.gradient_norms.pop(0)
        
        # Check for gradient explosion
        if total_norm > self.explosion_threshold:
            return ErrorType.GRADIENT_EXPLOSION
        
        return None
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics.
        
        Returns:
            Dict: Gradient statistics
        """
        if not self.gradient_norms:
            return {}
        
        return {
            'current_norm': self.gradient_norms[-1],
            'mean_norm': sum(self.gradient_norms) / len(self.gradient_norms),
            'max_norm': max(self.gradient_norms),
            'min_norm': min(self.gradient_norms),
        }
    
    def reset(self) -> None:
        """Reset gradient monitoring state."""
        self.gradient_norms.clear()
        self.error_count = 0


class LossMonitor:
    """Monitors loss values for anomalies."""
    
    def __init__(
        self,
        nan_check: bool = True,
        inf_check: bool = True,
        divergence_threshold: float = 100.0,
        history_size: int = 100
    ):
        """Initialize loss monitor.
        
        Args:
            nan_check: Whether to check for NaN loss
            inf_check: Whether to check for Inf loss
            divergence_threshold: Threshold for loss divergence detection
            history_size: Size of loss history
        """
        self.nan_check = nan_check
        self.inf_check = inf_check
        self.divergence_threshold = divergence_threshold
        self.history_size = history_size
        
        self.loss_history = []
        self.min_loss = float('inf')
        
    def check_loss(self, loss: torch.Tensor) -> Optional[ErrorType]:
        """Check loss value for anomalies.
        
        Args:
            loss: Loss tensor to check
            
        Returns:
            Optional[ErrorType]: Error type if anomaly detected, None otherwise
        """
        loss_value = loss.item()
        
        # Check for NaN
        if self.nan_check and torch.isnan(loss):
            return ErrorType.NAN_LOSS
        
        # Check for Inf
        if self.inf_check and torch.isinf(loss):
            return ErrorType.INF_LOSS
        
        # Update loss history
        self.loss_history.append(loss_value)
        if len(self.loss_history) > self.history_size:
            self.loss_history.pop(0)
        
        # Update minimum loss
        if loss_value < self.min_loss:
            self.min_loss = loss_value
        
        # Check for divergence (loss much higher than minimum)
        if loss_value > self.min_loss + self.divergence_threshold:
            return ErrorType.MODEL_DIVERGENCE
        
        return None
    
    def get_loss_stats(self) -> Dict[str, float]:
        """Get loss statistics.
        
        Returns:
            Dict: Loss statistics
        """
        if not self.loss_history:
            return {}
        
        return {
            'current_loss': self.loss_history[-1],
            'mean_loss': sum(self.loss_history) / len(self.loss_history),
            'min_loss': self.min_loss,
            'max_loss': max(self.loss_history),
        }
    
    def reset(self) -> None:
        """Reset loss monitoring state."""
        self.loss_history.clear()
        self.min_loss = float('inf')


class RecoveryStrategy:
    """Base class for recovery strategies."""
    
    def __init__(self, name: str):
        """Initialize recovery strategy.
        
        Args:
            name: Name of the recovery strategy
        """
        self.name = name
    
    def can_recover(self, error_event: ErrorEvent) -> bool:
        """Check if this strategy can recover from the error.
        
        Args:
            error_event: Error event to check
            
        Returns:
            bool: True if recovery is possible
        """
        raise NotImplementedError
    
    def recover(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        error_event: ErrorEvent,
        **kwargs
    ) -> bool:
        """Attempt to recover from the error.
        
        Args:
            model: Model to recover
            optimizer: Optimizer to recover
            scheduler: Scheduler to recover
            error_event: Error event information
            **kwargs: Additional recovery parameters
            
        Returns:
            bool: True if recovery was successful
        """
        raise NotImplementedError


class GradientClippingRecovery(RecoveryStrategy):
    """Recovery strategy using gradient clipping."""
    
    def __init__(self, max_norm: float = 1.0):
        """Initialize gradient clipping recovery.
        
        Args:
            max_norm: Maximum gradient norm for clipping
        """
        super().__init__("gradient_clipping")
        self.max_norm = max_norm
    
    def can_recover(self, error_event: ErrorEvent) -> bool:
        """Check if gradient clipping can recover from error."""
        return error_event.error_type in [
            ErrorType.GRADIENT_EXPLOSION,
            ErrorType.NAN_LOSS,
            ErrorType.INF_LOSS
        ]
    
    def recover(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        error_event: ErrorEvent,
        **kwargs
    ) -> bool:
        """Recover using gradient clipping."""
        try:
            # Zero gradients
            optimizer.zero_grad()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            
            return True
        except Exception:
            return False


class LearningRateReductionRecovery(RecoveryStrategy):
    """Recovery strategy using learning rate reduction."""
    
    def __init__(self, reduction_factor: float = 0.5, min_lr: float = 1e-8):
        """Initialize learning rate reduction recovery.
        
        Args:
            reduction_factor: Factor to reduce learning rate by
            min_lr: Minimum learning rate
        """
        super().__init__("learning_rate_reduction")
        self.reduction_factor = reduction_factor
        self.min_lr = min_lr
    
    def can_recover(self, error_event: ErrorEvent) -> bool:
        """Check if learning rate reduction can recover from error."""
        return error_event.error_type in [
            ErrorType.GRADIENT_EXPLOSION,
            ErrorType.MODEL_DIVERGENCE,
            ErrorType.NAN_LOSS
        ]
    
    def recover(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        error_event: ErrorEvent,
        **kwargs
    ) -> bool:
        """Recover using learning rate reduction."""
        try:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                new_lr = max(current_lr * self.reduction_factor, self.min_lr)
                param_group['lr'] = new_lr
            
            return True
        except Exception:
            return False


class CheckpointRecovery(RecoveryStrategy):
    """Recovery strategy using checkpoint restoration."""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        """Initialize checkpoint recovery.
        
        Args:
            checkpoint_manager: Checkpoint manager for loading checkpoints
        """
        super().__init__("checkpoint_recovery")
        self.checkpoint_manager = checkpoint_manager
    
    def can_recover(self, error_event: ErrorEvent) -> bool:
        """Check if checkpoint recovery is possible."""
        return error_event.error_type in [
            ErrorType.GRADIENT_EXPLOSION,
            ErrorType.MODEL_DIVERGENCE,
            ErrorType.NAN_LOSS,
            ErrorType.INF_LOSS,
            ErrorType.CHECKPOINT_CORRUPTION
        ]
    
    def recover(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        error_event: ErrorEvent,
        **kwargs
    ) -> bool:
        """Recover using checkpoint restoration."""
        try:
            # Try to load the latest checkpoint
            result = self.checkpoint_manager.load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                strict=False  # Allow partial loading
            )
            
            return result is not None
        except Exception:
            return False


class ModelReinitializationRecovery(RecoveryStrategy):
    """Recovery strategy using partial model reinitialization."""
    
    def __init__(self, reinit_layers: Optional[List[str]] = None):
        """Initialize model reinitialization recovery.
        
        Args:
            reinit_layers: List of layer names to reinitialize (None for all)
        """
        super().__init__("model_reinitialization")
        self.reinit_layers = reinit_layers
    
    def can_recover(self, error_event: ErrorEvent) -> bool:
        """Check if model reinitialization can recover from error."""
        return error_event.error_type in [
            ErrorType.NAN_LOSS,
            ErrorType.INF_LOSS,
            ErrorType.MODEL_DIVERGENCE
        ]
    
    def recover(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        error_event: ErrorEvent,
        **kwargs
    ) -> bool:
        """Recover using model reinitialization."""
        try:
            if self.reinit_layers is None:
                # Reinitialize all parameters
                for module in model.modules():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
            else:
                # Reinitialize specific layers
                for name, module in model.named_modules():
                    if name in self.reinit_layers and hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
            
            # Reset optimizer state
            optimizer.state.clear()
            
            return True
        except Exception:
            return False


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[TrainingLogger] = None,
        max_recovery_attempts: int = 3,
        recovery_strategies: Optional[List[RecoveryStrategy]] = None
    ):
        """Initialize error handler.
        
        Args:
            checkpoint_manager: Checkpoint manager for recovery
            logger: Logger for error reporting
            max_recovery_attempts: Maximum number of recovery attempts
            recovery_strategies: List of recovery strategies to use
        """
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.max_recovery_attempts = max_recovery_attempts
        
        # Initialize monitors
        self.gradient_monitor = GradientMonitor()
        self.loss_monitor = LossMonitor()
        
        # Initialize recovery strategies
        if recovery_strategies is None:
            self.recovery_strategies = self._create_default_strategies()
        else:
            self.recovery_strategies = recovery_strategies
        
        # Error tracking
        self.error_history: List[ErrorEvent] = []
        self.recovery_attempts = 0
        
    def _create_default_strategies(self) -> List[RecoveryStrategy]:
        """Create default recovery strategies.
        
        Returns:
            List[RecoveryStrategy]: Default recovery strategies
        """
        strategies = [
            GradientClippingRecovery(max_norm=1.0),
            LearningRateReductionRecovery(reduction_factor=0.5),
        ]
        
        if self.checkpoint_manager:
            strategies.append(CheckpointRecovery(self.checkpoint_manager))
        
        strategies.append(ModelReinitializationRecovery())
        
        return strategies
    
    def check_for_errors(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        step: int,
        epoch: int
    ) -> Optional[ErrorEvent]:
        """Check for training errors.
        
        Args:
            model: Model to check
            loss: Current loss value
            step: Current training step
            epoch: Current epoch
            
        Returns:
            Optional[ErrorEvent]: Error event if error detected, None otherwise
        """
        # Check gradients
        gradient_error = self.gradient_monitor.check_gradients(model)
        if gradient_error:
            return self._create_error_event(gradient_error, step, epoch)
        
        # Check loss
        loss_error = self.loss_monitor.check_loss(loss)
        if loss_error:
            return self._create_error_event(loss_error, step, epoch)
        
        return None
    
    def handle_error(
        self,
        error_event: ErrorEvent,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        **kwargs
    ) -> bool:
        """Handle training error with recovery attempts.
        
        Args:
            error_event: Error event to handle
            model: Model to recover
            optimizer: Optimizer to recover
            scheduler: Scheduler to recover
            **kwargs: Additional recovery parameters
            
        Returns:
            bool: True if recovery was successful
        """
        # Log error
        if self.logger:
            self.logger.log_error(
                Exception(error_event.error_message),
                context=f"Step {error_event.step}, Error type: {error_event.error_type.value}"
            )
        
        # Add to error history
        self.error_history.append(error_event)
        
        # Check if we've exceeded max recovery attempts
        if self.recovery_attempts >= self.max_recovery_attempts:
            if self.logger:
                self.logger.logger.error(
                    f"Maximum recovery attempts ({self.max_recovery_attempts}) exceeded"
                )
            return False
        
        # Try recovery strategies
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error_event):
                try:
                    success = strategy.recover(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        error_event=error_event,
                        **kwargs
                    )
                    
                    if success:
                        error_event.recovery_attempted = True
                        error_event.recovery_successful = True
                        error_event.recovery_method = strategy.name
                        
                        if self.logger:
                            self.logger.logger.info(
                                f"Successfully recovered from {error_event.error_type.value} "
                                f"using {strategy.name}"
                            )
                        
                        self.recovery_attempts += 1
                        return True
                    
                except Exception as e:
                    if self.logger:
                        self.logger.logger.warning(
                            f"Recovery strategy {strategy.name} failed: {str(e)}"
                        )
        
        # No recovery strategy succeeded
        error_event.recovery_attempted = True
        error_event.recovery_successful = False
        
        if self.logger:
            self.logger.logger.error(
                f"Failed to recover from {error_event.error_type.value}"
            )
        
        return False
    
    def _create_error_event(
        self,
        error_type: ErrorType,
        step: int,
        epoch: int
    ) -> ErrorEvent:
        """Create error event.
        
        Args:
            error_type: Type of error
            step: Current step
            epoch: Current epoch
            
        Returns:
            ErrorEvent: Created error event
        """
        return ErrorEvent(
            error_type=error_type,
            step=step,
            epoch=epoch,
            error_message=f"{error_type.value} detected at step {step}",
            traceback_str=traceback.format_exc(),
            timestamp=time.time()
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics.
        
        Returns:
            Dict: Error statistics
        """
        if not self.error_history:
            return {}
        
        error_counts = {}
        recovery_success_rate = 0
        
        for event in self.error_history:
            error_type = event.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            if event.recovery_attempted and event.recovery_successful:
                recovery_success_rate += 1
        
        recovery_success_rate = recovery_success_rate / len(self.error_history)
        
        return {
            'total_errors': len(self.error_history),
            'error_counts': error_counts,
            'recovery_success_rate': recovery_success_rate,
            'recovery_attempts': self.recovery_attempts,
            'gradient_stats': self.gradient_monitor.get_gradient_stats(),
            'loss_stats': self.loss_monitor.get_loss_stats(),
        }
    
    def reset(self) -> None:
        """Reset error handler state."""
        self.error_history.clear()
        self.recovery_attempts = 0
        self.gradient_monitor.reset()
        self.loss_monitor.reset()


def create_error_handler(
    checkpoint_manager: Optional[CheckpointManager] = None,
    logger: Optional[TrainingLogger] = None,
    gradient_clip_norm: float = 1.0,
    lr_reduction_factor: float = 0.5,
    max_recovery_attempts: int = 3
) -> ErrorHandler:
    """Create error handler with default configuration.
    
    Args:
        checkpoint_manager: Checkpoint manager for recovery
        logger: Logger for error reporting
        gradient_clip_norm: Maximum gradient norm for clipping
        lr_reduction_factor: Factor for learning rate reduction
        max_recovery_attempts: Maximum recovery attempts
        
    Returns:
        ErrorHandler: Configured error handler
    """
    strategies = [
        GradientClippingRecovery(max_norm=gradient_clip_norm),
        LearningRateReductionRecovery(reduction_factor=lr_reduction_factor),
    ]
    
    if checkpoint_manager:
        strategies.append(CheckpointRecovery(checkpoint_manager))
    
    return ErrorHandler(
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        max_recovery_attempts=max_recovery_attempts,
        recovery_strategies=strategies
    )