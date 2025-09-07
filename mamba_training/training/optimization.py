"""Optimization and learning rate scheduling for Mamba training."""

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any, Optional, Union
import logging

from ..config import TrainingConfig


logger = logging.getLogger(__name__)


class OptimizationManager:
    """Manages optimization, learning rate scheduling, and gradient operations."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        total_steps: Optional[int] = None
    ):
        """Initialize optimization manager.
        
        Args:
            model: The model to optimize
            config: Training configuration
            total_steps: Total number of training steps for LR scheduling
        """
        self.config = config
        self.model = model
        self.total_steps = total_steps
        self.current_step = 0
        self.accumulated_steps = 0
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler() if total_steps and total_steps > 0 else None
        
        # Initialize gradient scaler for mixed precision
        if config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        logger.info(f"Initialized OptimizationManager with lr={config.learning_rate}, "
                   f"weight_decay={config.weight_decay}, warmup_steps={config.warmup_steps}")
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay.
        
        Returns:
            AdamW optimizer instance
        """
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Don't apply weight decay to bias terms and layer norm parameters
            if any(nd in name for nd in ['bias', 'norm', 'ln']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                'params': decay_params,
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),  # Standard values for language models
            eps=1e-8,
        )
    
    def _create_scheduler(self) -> LambdaLR:
        """Create cosine learning rate scheduler with warmup.
        
        Returns:
            Learning rate scheduler
        """
        def lr_lambda(current_step: int) -> float:
            """Compute learning rate multiplier for given step."""
            if current_step < self.config.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.config.warmup_steps))
            
            # Cosine annealing after warmup
            progress = float(current_step - self.config.warmup_steps)
            total_progress = float(self.total_steps - self.config.warmup_steps)
            
            if total_progress <= 0:
                return 1.0
            
            return 0.5 * (1.0 + math.cos(math.pi * progress / total_progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """Perform optimization step with gradient accumulation and clipping.
        
        Args:
            loss: Loss tensor to backpropagate
            
        Returns:
            Dictionary containing optimization metrics
        """
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with optional mixed precision
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        self.accumulated_steps += 1
        
        # Only step optimizer after accumulating enough gradients
        if self.accumulated_steps >= self.config.gradient_accumulation_steps:
            return self._optimizer_step()
        
        return {
            'learning_rate': self.get_current_lr(),
            'grad_norm': 0.0,
            'optimizer_step': False,
        }
    
    def _optimizer_step(self) -> Dict[str, float]:
        """Perform actual optimizer step with gradient clipping.
        
        Returns:
            Dictionary containing optimization metrics
        """
        # Unscale gradients for clipping if using mixed precision
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Compute gradient norm before clipping
        grad_norm = self._compute_grad_norm()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Reset accumulation counter
        self.accumulated_steps = 0
        self.current_step += 1
        
        return {
            'learning_rate': self.get_current_lr(),
            'grad_norm': grad_norm,
            'optimizer_step': True,
        }
    
    def _compute_grad_norm(self) -> float:
        """Compute the L2 norm of gradients.
        
        Returns:
            Gradient norm as float
        """
        total_norm = 0.0
        param_count = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count == 0:
            return 0.0
        
        return total_norm ** 0.5
    
    def get_current_lr(self) -> float:
        """Get current learning rate.
        
        Returns:
            Current learning rate
        """
        return self.optimizer.param_groups[0]['lr']
    
    def get_optimizer_state(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing.
        
        Returns:
            Dictionary containing optimizer state
        """
        state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_step': self.current_step,
            'accumulated_steps': self.accumulated_steps,
        }
        
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
        
        return state
    
    def load_optimizer_state(self, state: Dict[str, Any]) -> None:
        """Load optimizer state from checkpoint.
        
        Args:
            state: Dictionary containing optimizer state
        """
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.current_step = state.get('current_step', 0)
        self.accumulated_steps = state.get('accumulated_steps', 0)
        
        if self.scheduler is not None and 'scheduler_state_dict' in state:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in state:
            self.scaler.load_state_dict(state['scaler_state_dict'])
        
        logger.info(f"Loaded optimizer state at step {self.current_step}")
    
    def zero_grad(self) -> None:
        """Clear gradients."""
        self.optimizer.zero_grad()
    
    def get_lr_schedule_preview(self, num_steps: int = 1000) -> Dict[int, float]:
        """Get preview of learning rate schedule.
        
        Args:
            num_steps: Number of steps to preview
            
        Returns:
            Dictionary mapping step to learning rate
        """
        if self.scheduler is None:
            return {i: self.config.learning_rate for i in range(num_steps)}
        
        if self.total_steps is None or self.total_steps <= 0:
            return {}
        
        # Create temporary scheduler for preview
        temp_optimizer = AdamW([torch.tensor(0.0, requires_grad=True)], lr=self.config.learning_rate)
        
        def lr_lambda(current_step: int) -> float:
            """Compute learning rate multiplier for given step."""
            if current_step < self.config.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.config.warmup_steps))
            
            # Cosine annealing after warmup
            progress = float(current_step - self.config.warmup_steps)
            total_progress = float(self.total_steps - self.config.warmup_steps)
            
            if total_progress <= 0:
                return 1.0
            
            return 0.5 * (1.0 + math.cos(math.pi * progress / total_progress))
        
        schedule = {}
        for step in range(min(num_steps, self.total_steps)):
            schedule[step] = self.config.learning_rate * lr_lambda(step)
        
        return schedule


class LearningRateScheduler:
    """Standalone learning rate scheduler utilities."""
    
    @staticmethod
    def cosine_with_warmup(
        current_step: int,
        warmup_steps: int,
        total_steps: int,
        base_lr: float = 1.0,
        min_lr_ratio: float = 0.0
    ) -> float:
        """Cosine learning rate schedule with linear warmup.
        
        Args:
            current_step: Current training step
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            base_lr: Base learning rate multiplier
            min_lr_ratio: Minimum learning rate as ratio of base_lr
            
        Returns:
            Learning rate multiplier
        """
        if current_step < warmup_steps:
            # Linear warmup
            return base_lr * float(current_step) / float(max(1, warmup_steps))
        
        # Cosine annealing
        progress = float(current_step - warmup_steps)
        total_progress = float(total_steps - warmup_steps)
        
        if total_progress <= 0:
            return base_lr
        
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress / total_progress))
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor)
    
    @staticmethod
    def linear_with_warmup(
        current_step: int,
        warmup_steps: int,
        total_steps: int,
        base_lr: float = 1.0
    ) -> float:
        """Linear learning rate schedule with linear warmup.
        
        Args:
            current_step: Current training step
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            base_lr: Base learning rate multiplier
            
        Returns:
            Learning rate multiplier
        """
        if current_step < warmup_steps:
            # Linear warmup
            return base_lr * float(current_step) / float(max(1, warmup_steps))
        
        # Linear decay
        progress = float(current_step - warmup_steps)
        total_progress = float(total_steps - warmup_steps)
        
        if total_progress <= 0:
            return base_lr
        
        return base_lr * (1.0 - progress / total_progress)


def create_optimization_manager(
    model: nn.Module,
    config: TrainingConfig,
    total_steps: Optional[int] = None
) -> OptimizationManager:
    """Factory function to create OptimizationManager.
    
    Args:
        model: Model to optimize
        config: Training configuration
        total_steps: Total training steps for LR scheduling
        
    Returns:
        OptimizationManager instance
    """
    return OptimizationManager(model, config, total_steps)