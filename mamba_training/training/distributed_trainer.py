"""Distributed training engine for Mamba models."""

import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast

from ..config import Config, TrainingConfig
from .optimization import OptimizationManager


logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Distributed training engine with DDP/FSDP support."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        """Initialize distributed trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Initialize distributed training
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.is_distributed = self.world_size > 1
        
        if self.is_distributed:
            self._init_distributed()
        
        # Setup device
        self.device = self._setup_device()
        
        # Wrap model for distributed training
        self.model = self._wrap_model(model)
        
        # Calculate total training steps
        self.total_steps = self._calculate_total_steps()
        
        # Initialize optimization manager
        self.optimizer_manager = OptimizationManager(
            self.model,
            config.training,
            total_steps=self.total_steps
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_metrics = []
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        logger.info(f"Initialized DistributedTrainer on rank {self.rank}/{self.world_size}")
        logger.info(f"Total training steps: {self.total_steps}")
    
    def _init_distributed(self) -> None:
        """Initialize distributed training backend."""
        if not dist.is_initialized():
            # Use NCCL backend for GPU training
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend)
        
        # Set device for current process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.local_rank}')
        else:
            device = torch.device('cpu')
        
        logger.info(f"Using device: {device}")
        return device
    
    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training.
        
        Args:
            model: Model to wrap
            
        Returns:
            Wrapped model
        """
        model = model.to(self.device)
        
        if not self.is_distributed:
            return model
        
        # Use FSDP for large models, DDP for smaller ones
        if self._should_use_fsdp(model):
            logger.info("Using FSDP for distributed training")
            
            # Define auto wrap policy for transformer blocks
            auto_wrap_policy = transformer_auto_wrap_policy
            
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=self._get_fsdp_mixed_precision(),
                device_id=self.local_rank,
            )
        else:
            logger.info("Using DDP for distributed training")
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        return model
    
    def _should_use_fsdp(self, model: nn.Module) -> bool:
        """Determine whether to use FSDP based on model size."""
        # Use FSDP for models with more than 1B parameters
        total_params = sum(p.numel() for p in model.parameters())
        return total_params > 1e9
    
    def _get_fsdp_mixed_precision(self):
        """Get FSDP mixed precision policy."""
        if not self.config.training.use_mixed_precision:
            return None
        
        from torch.distributed.fsdp import MixedPrecision
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    
    def _calculate_total_steps(self) -> int:
        """Calculate total number of training steps."""
        steps_per_epoch = len(self.train_dataloader) // self.config.training.gradient_accumulation_steps
        return steps_per_epoch * self.config.training.num_epochs
    
    def train(self) -> Dict[str, Any]:
        """Run complete training loop.
        
        Returns:
            Training metrics and final results
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Training epoch
            train_metrics = self.train_epoch()
            
            # Validation epoch
            val_metrics = {}
            if self.val_dataloader is not None:
                val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'global_step': self.global_step,
            }
            
            self.training_metrics.append(epoch_metrics)
            
            # Save checkpoint
            if self.rank == 0 and ((epoch + 1) % self._get_save_frequency() == 0 or epoch == self.config.training.num_epochs - 1):
                self.save_checkpoint(epoch_metrics)
            
            # Log progress
            if self.rank == 0:
                self._log_epoch_metrics(epoch_metrics)
        
        total_time = time.time() - start_time
        
        # Final checkpoint
        if self.rank == 0:
            final_metrics = {
                'epoch': self.current_epoch,
                'train': train_metrics,
                'val': val_metrics,
                'global_step': self.global_step,
                'total_training_time': total_time,
            }
            self.save_checkpoint(final_metrics, is_final=True)
        
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return {
            'total_time': total_time,
            'final_metrics': epoch_metrics,
            'training_history': self.training_metrics,
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Convert batch to dict format if needed
            if isinstance(batch, (list, tuple)):
                batch = {'input_ids': batch[0], 'labels': batch[1] if len(batch) > 1 else None}
            elif hasattr(batch, 'input_ids'):  # BatchedSample object
                batch = {
                    'input_ids': batch.input_ids,
                    'attention_mask': batch.attention_mask,
                    'labels': batch.labels
                }
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Accumulate metrics
            if step_metrics['optimizer_step']:
                total_loss += step_metrics['loss']
                total_tokens += step_metrics.get('num_tokens', 0)
                num_batches += 1
                
                # Log step metrics
                if self.global_step % self._get_log_frequency() == 0 and self.rank == 0:
                    self._log_step_metrics(step_metrics)
                
                # Validation during training
                if (self.global_step % self.config.training.eval_steps == 0 and 
                    self.val_dataloader is not None):
                    val_metrics = self.validate()
                    if self.rank == 0:
                        logger.info(f"Step {self.global_step} validation: {val_metrics}")
        
        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        tokens_per_sec = total_tokens / epoch_time if epoch_time > 0 else 0
        
        # Synchronize metrics across processes
        if self.is_distributed:
            avg_loss = self._all_reduce_metric(avg_loss)
            tokens_per_sec = self._all_reduce_metric(tokens_per_sec)
        
        return {
            'loss': avg_loss,
            'tokens_per_second': tokens_per_sec,
            'epoch_time': epoch_time,
            'num_batches': num_batches,
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform single training step.
        
        Args:
            batch: Input batch
            
        Returns:
            Step metrics
        """
        # Forward pass with mixed precision
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with autocast(device_type=device_type, enabled=self.config.training.use_mixed_precision):
            outputs = self.model(input_ids=batch['input_ids'], return_dict=True)
            
            # Compute loss manually since MambaModel doesn't compute loss internally
            if batch['labels'] is not None:
                logits = outputs.logits
                # Shift labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch['labels'][..., 1:].contiguous()
                
                # Compute cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                # If no labels, use a dummy loss for testing
                loss = outputs.logits.sum() * 0
        
        # Optimization step
        step_metrics = self.optimizer_manager.step(loss)
        
        # Update global step counter
        if step_metrics['optimizer_step']:
            self.global_step += 1
        
        # Add loss and token count to metrics
        step_metrics.update({
            'loss': loss.item(),
            'num_tokens': batch.get('input_ids', torch.tensor([])).numel(),
            'global_step': self.global_step,
        })
        
        return step_metrics
    
    def validate(self) -> Dict[str, float]:
        """Run validation loop.
        
        Returns:
            Validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Convert batch to dict format if needed
                if isinstance(batch, (list, tuple)):
                    batch = {'input_ids': batch[0], 'labels': batch[1] if len(batch) > 1 else None}
                elif hasattr(batch, 'input_ids'):  # BatchedSample object
                    batch = {
                        'input_ids': batch.input_ids,
                        'attention_mask': batch.attention_mask,
                        'labels': batch.labels
                    }
                
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with autocast(device_type=device_type, enabled=self.config.training.use_mixed_precision):
                    outputs = self.model(input_ids=batch['input_ids'], return_dict=True)
                    
                    # Compute loss manually
                    if batch['labels'] is not None:
                        logits = outputs.logits
                        # Shift labels for next-token prediction
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = batch['labels'][..., 1:].contiguous()
                        
                        # Compute cross-entropy loss
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    else:
                        loss = outputs.logits.sum() * 0
                
                total_loss += loss.item()
                total_tokens += batch.get('input_ids', torch.tensor([])).numel()
                num_batches += 1
        
        # Calculate metrics
        avg_loss = total_loss / max(num_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Synchronize metrics across processes
        if self.is_distributed:
            avg_loss = self._all_reduce_metric(avg_loss)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Update best validation loss
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        
        self.model.train()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'num_batches': num_batches,
            'is_best': avg_loss == self.best_val_loss,
        }
    
    def save_checkpoint(
        self,
        metrics: Dict[str, Any],
        is_final: bool = False
    ) -> None:
        """Save training checkpoint.
        
        Args:
            metrics: Current training metrics
            is_final: Whether this is the final checkpoint
        """
        if self.rank != 0:
            return
        
        checkpoint_name = 'final_checkpoint.pt' if is_final else f'checkpoint_epoch_{self.current_epoch}.pt'
        checkpoint_path = self.output_dir / checkpoint_name
        
        # Get model state dict
        if isinstance(self.model, (DDP, FSDP)):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state': self.optimizer_manager.get_optimizer_state(),
            'config': self.config,
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'training_metrics': self.training_metrics,
        }
        
        # Save checkpoint atomically
        temp_path = checkpoint_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = self.output_dir / 'latest_checkpoint.pt'
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(checkpoint_path.name)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        if isinstance(self.model, (DDP, FSDP)):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer_manager.load_optimizer_state(checkpoint['optimizer_state'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_metrics = checkpoint.get('training_metrics', [])
        
        logger.info(f"Resumed training from epoch {self.current_epoch}, step {self.global_step}")
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to training device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
    
    def _all_reduce_metric(self, metric: float) -> float:
        """All-reduce metric across processes."""
        if not self.is_distributed:
            return metric
        
        tensor = torch.tensor(metric, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return (tensor / self.world_size).item()
    
    def _get_save_frequency(self) -> int:
        """Get checkpoint save frequency in epochs."""
        steps_per_epoch = len(self.train_dataloader) // self.config.training.gradient_accumulation_steps
        if steps_per_epoch == 0:
            return 1  # Save every epoch for very small datasets
        return max(1, self.config.training.save_steps // steps_per_epoch)
    
    def _get_log_frequency(self) -> int:
        """Get logging frequency in steps."""
        return min(100, max(1, len(self.train_dataloader) // 10))
    
    def _log_step_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log step-level metrics."""
        logger.info(
            f"Step {metrics['global_step']}: "
            f"loss={metrics['loss']:.4f}, "
            f"lr={metrics['learning_rate']:.2e}, "
            f"grad_norm={metrics['grad_norm']:.4f}"
        )
    
    def _log_epoch_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log epoch-level metrics."""
        train_metrics = metrics['train']
        val_metrics = metrics.get('val', {})
        
        log_msg = (
            f"Epoch {metrics['epoch']}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"tokens/sec={train_metrics['tokens_per_second']:.0f}"
        )
        
        if val_metrics:
            log_msg += f", val_loss={val_metrics['loss']:.4f}, val_ppl={val_metrics['perplexity']:.2f}"
        
        logger.info(log_msg)
    
    def cleanup(self) -> None:
        """Cleanup distributed training resources."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def create_distributed_trainer(
    model: nn.Module,
    config: Config,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    resume_from_checkpoint: Optional[str] = None
) -> DistributedTrainer:
    """Factory function to create DistributedTrainer.
    
    Args:
        model: Model to train
        config: Training configuration
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        DistributedTrainer instance
    """
    return DistributedTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        resume_from_checkpoint=resume_from_checkpoint
    )


def setup_distributed_sampler(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """Setup distributed data loader with sampler.
    
    Args:
        dataset: Dataset to sample from
        batch_size: Batch size per process
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        Tuple of (DataLoader, DistributedSampler or None)
    """
    sampler = None
    
    # Use distributed sampler if in distributed mode
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last
        )
        shuffle = False  # Sampler handles shuffling
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        num_workers=4,
    )
    
    return dataloader, sampler