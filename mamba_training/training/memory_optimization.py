"""Memory optimization utilities for Mamba training."""

import gc
import logging
import psutil
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, List, Tuple
from contextlib import contextmanager
from functools import wraps
import time

from ..config import TrainingConfig


logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Memory profiling and monitoring utilities."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize memory profiler.
        
        Args:
            device: Device to monitor (defaults to current device)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_cuda = self.device.type == 'cuda'
        self.memory_history = []
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        stats = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        stats['system'] = {
            'total': system_memory.total,
            'available': system_memory.available,
            'used': system_memory.used,
            'percent': system_memory.percent,
        }
        
        # GPU memory (if available)
        if self.is_cuda and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats(self.device)
            stats['gpu'] = {
                'allocated': torch.cuda.memory_allocated(self.device),
                'reserved': torch.cuda.memory_reserved(self.device),
                'max_allocated': torch.cuda.max_memory_allocated(self.device),
                'max_reserved': torch.cuda.max_memory_reserved(self.device),
            }
            
            # Add detailed GPU stats
            stats['gpu'].update({
                'active_bytes': gpu_memory.get('active_bytes.all.current', 0),
                'inactive_split_bytes': gpu_memory.get('inactive_split_bytes.all.current', 0),
                'allocated_bytes': gpu_memory.get('allocated_bytes.all.current', 0),
                'reserved_bytes': gpu_memory.get('reserved_bytes.all.current', 0),
            })
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        stats['process'] = {
            'rss': process_memory.rss,  # Resident Set Size
            'vms': process_memory.vms,  # Virtual Memory Size
        }
        
        return stats
    
    def log_memory_stats(self, prefix: str = "") -> None:
        """Log current memory statistics.
        
        Args:
            prefix: Prefix for log message
        """
        stats = self.get_memory_stats()
        
        # Format system memory
        system = stats['system']
        system_msg = f"System: {system['used'] / 1e9:.2f}GB / {system['total'] / 1e9:.2f}GB ({system['percent']:.1f}%)"
        
        # Format GPU memory
        gpu_msg = ""
        if 'gpu' in stats:
            gpu = stats['gpu']
            gpu_msg = f", GPU: {gpu['allocated'] / 1e9:.2f}GB allocated, {gpu['reserved'] / 1e9:.2f}GB reserved"
        
        # Format process memory
        process = stats['process']
        process_msg = f", Process: {process['rss'] / 1e9:.2f}GB RSS"
        
        logger.info(f"{prefix}Memory - {system_msg}{gpu_msg}{process_msg}")
    
    def record_memory_snapshot(self, tag: str) -> None:
        """Record memory snapshot with tag.
        
        Args:
            tag: Tag to identify this snapshot
        """
        stats = self.get_memory_stats()
        stats['tag'] = tag
        stats['timestamp'] = time.time()
        self.memory_history.append(stats)
    
    def get_memory_diff(self, start_tag: str, end_tag: str) -> Dict[str, Any]:
        """Get memory difference between two snapshots.
        
        Args:
            start_tag: Tag of starting snapshot
            end_tag: Tag of ending snapshot
            
        Returns:
            Dictionary containing memory differences
        """
        start_snapshot = None
        end_snapshot = None
        
        for snapshot in self.memory_history:
            if snapshot['tag'] == start_tag:
                start_snapshot = snapshot
            elif snapshot['tag'] == end_tag:
                end_snapshot = snapshot
        
        if start_snapshot is None or end_snapshot is None:
            raise ValueError(f"Could not find snapshots for tags: {start_tag}, {end_tag}")
        
        diff = {}
        
        # System memory diff
        diff['system'] = {
            'used_diff': end_snapshot['system']['used'] - start_snapshot['system']['used'],
            'percent_diff': end_snapshot['system']['percent'] - start_snapshot['system']['percent'],
        }
        
        # GPU memory diff
        if 'gpu' in start_snapshot and 'gpu' in end_snapshot:
            diff['gpu'] = {
                'allocated_diff': end_snapshot['gpu']['allocated'] - start_snapshot['gpu']['allocated'],
                'reserved_diff': end_snapshot['gpu']['reserved'] - start_snapshot['gpu']['reserved'],
            }
        
        # Process memory diff
        diff['process'] = {
            'rss_diff': end_snapshot['process']['rss'] - start_snapshot['process']['rss'],
            'vms_diff': end_snapshot['process']['vms'] - start_snapshot['process']['vms'],
        }
        
        diff['time_diff'] = end_snapshot['timestamp'] - start_snapshot['timestamp']
        
        return diff
    
    def clear_history(self) -> None:
        """Clear memory history."""
        self.memory_history.clear()
    
    @contextmanager
    def profile_memory(self, tag: str):
        """Context manager for profiling memory usage.
        
        Args:
            tag: Tag for this profiling session
        """
        start_tag = f"{tag}_start"
        end_tag = f"{tag}_end"
        
        self.record_memory_snapshot(start_tag)
        try:
            yield
        finally:
            self.record_memory_snapshot(end_tag)
            diff = self.get_memory_diff(start_tag, end_tag)
            
            # Log memory usage
            system_diff = diff['system']['used_diff'] / 1e9
            gpu_diff = diff.get('gpu', {}).get('allocated_diff', 0) / 1e9
            process_diff = diff['process']['rss_diff'] / 1e9
            
            logger.info(
                f"Memory usage for {tag}: "
                f"System: {system_diff:+.2f}GB, "
                f"GPU: {gpu_diff:+.2f}GB, "
                f"Process: {process_diff:+.2f}GB, "
                f"Time: {diff['time_diff']:.2f}s"
            )


class GradientCheckpointing:
    """Gradient checkpointing utilities for memory optimization."""
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module) -> None:
        """Enable gradient checkpointing for supported modules.
        
        Args:
            model: Model to enable gradient checkpointing for
        """
        def enable_checkpointing_recursive(module):
            # Enable gradient checkpointing for MambaBlock
            if hasattr(module, '__class__') and 'MambaBlock' in module.__class__.__name__:
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
                else:
                    # Wrap the forward method with checkpointing
                    original_forward = module.forward
                    
                    def checkpointed_forward(*args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(
                            original_forward, *args, **kwargs, use_reentrant=False
                        )
                    
                    module.forward = checkpointed_forward
                    module.gradient_checkpointing = True
                
                logger.info(f"Enabled gradient checkpointing for {module.__class__.__name__}")
            
            # Recursively apply to child modules
            for child in module.children():
                enable_checkpointing_recursive(child)
        
        enable_checkpointing_recursive(model)
    
    @staticmethod
    def disable_gradient_checkpointing(model: nn.Module) -> None:
        """Disable gradient checkpointing for all modules.
        
        Args:
            model: Model to disable gradient checkpointing for
        """
        def disable_checkpointing_recursive(module):
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = False
                logger.info(f"Disabled gradient checkpointing for {module.__class__.__name__}")
            
            for child in module.children():
                disable_checkpointing_recursive(child)
        
        disable_checkpointing_recursive(model)


class DynamicBatchSizer:
    """Dynamic batch size adjustment for OOM handling."""
    
    def __init__(
        self,
        initial_batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        reduction_factor: float = 0.5,
        increase_factor: float = 1.2,
        memory_threshold: float = 0.9,
        patience: int = 3
    ):
        """Initialize dynamic batch sizer.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            reduction_factor: Factor to reduce batch size on OOM
            increase_factor: Factor to increase batch size when memory allows
            memory_threshold: Memory usage threshold to trigger reduction
            patience: Number of successful steps before attempting increase
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size or initial_batch_size * 4
        self.reduction_factor = reduction_factor
        self.increase_factor = increase_factor
        self.memory_threshold = memory_threshold
        self.patience = patience
        
        self.successful_steps = 0
        self.oom_count = 0
        self.profiler = MemoryProfiler()
        
        logger.info(f"Initialized DynamicBatchSizer with batch_size={initial_batch_size}")
    
    def handle_oom(self) -> int:
        """Handle out-of-memory error by reducing batch size.
        
        Returns:
            New batch size
        """
        old_batch_size = self.current_batch_size
        self.current_batch_size = max(
            self.min_batch_size,
            int(self.current_batch_size * self.reduction_factor)
        )
        self.oom_count += 1
        self.successful_steps = 0
        
        logger.warning(
            f"OOM detected (#{self.oom_count}). "
            f"Reducing batch size: {old_batch_size} -> {self.current_batch_size}"
        )
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return self.current_batch_size
    
    def step_successful(self) -> int:
        """Record successful training step and potentially increase batch size.
        
        Returns:
            Current batch size (potentially increased)
        """
        self.successful_steps += 1
        
        # Check if we can increase batch size
        if (self.successful_steps >= self.patience and 
            self.current_batch_size < self.max_batch_size):
            
            # Check memory usage
            stats = self.profiler.get_memory_stats()
            
            memory_usage = 0.0
            if 'gpu' in stats and torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_usage = stats['gpu']['allocated'] / total_memory
            else:
                memory_usage = stats['system']['percent'] / 100.0
            
            if memory_usage < self.memory_threshold:
                old_batch_size = self.current_batch_size
                self.current_batch_size = min(
                    self.max_batch_size,
                    int(self.current_batch_size * self.increase_factor)
                )
                self.successful_steps = 0
                
                logger.info(
                    f"Memory usage low ({memory_usage:.1%}). "
                    f"Increasing batch size: {old_batch_size} -> {self.current_batch_size}"
                )
        
        return self.current_batch_size
    
    def get_current_batch_size(self) -> int:
        """Get current batch size.
        
        Returns:
            Current batch size
        """
        return self.current_batch_size
    
    def reset(self) -> None:
        """Reset counters and statistics."""
        self.successful_steps = 0
        self.oom_count = 0


def memory_efficient_forward(func: Callable) -> Callable:
    """Decorator for memory-efficient forward passes.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with memory optimizations
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Clear cache before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
        
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM in {func.__name__}: {e}")
                # Clear cache and try again with smaller inputs if possible
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                raise
            else:
                raise
    
    return wrapper


class MemoryOptimizer:
    """Main memory optimization coordinator."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize memory optimizer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.profiler = MemoryProfiler()
        self.dynamic_batcher = DynamicBatchSizer(
            initial_batch_size=config.batch_size,
            min_batch_size=1,
            max_batch_size=config.batch_size * 4
        )
        
        # Memory optimization settings
        self.gradient_checkpointing_enabled = config.gradient_checkpointing
        self.mixed_precision_enabled = config.use_mixed_precision
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations to model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        if self.gradient_checkpointing_enabled:
            GradientCheckpointing.enable_gradient_checkpointing(model)
        
        return model
    
    def handle_oom_error(self, error: RuntimeError) -> Dict[str, Any]:
        """Handle out-of-memory error.
        
        Args:
            error: The OOM error
            
        Returns:
            Dictionary with recovery information
        """
        logger.error(f"Out of memory error: {error}")
        
        # Reduce batch size
        new_batch_size = self.dynamic_batcher.handle_oom()
        
        # Clear memory
        self.clear_memory()
        
        # Log memory stats after cleanup
        self.profiler.log_memory_stats("After OOM cleanup")
        
        return {
            'new_batch_size': new_batch_size,
            'oom_count': self.dynamic_batcher.oom_count,
            'action': 'batch_size_reduced'
        }
    
    def step_completed(self) -> Dict[str, Any]:
        """Record successful training step.
        
        Returns:
            Dictionary with step information
        """
        new_batch_size = self.dynamic_batcher.step_successful()
        
        return {
            'batch_size': new_batch_size,
            'successful_steps': self.dynamic_batcher.successful_steps,
        }
    
    def clear_memory(self) -> None:
        """Clear memory caches."""
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report.
        
        Returns:
            Dictionary containing memory report
        """
        stats = self.profiler.get_memory_stats()
        
        report = {
            'memory_stats': stats,
            'current_batch_size': self.dynamic_batcher.get_current_batch_size(),
            'oom_count': self.dynamic_batcher.oom_count,
            'successful_steps': self.dynamic_batcher.successful_steps,
            'gradient_checkpointing': self.gradient_checkpointing_enabled,
            'mixed_precision': self.mixed_precision_enabled,
        }
        
        return report
    
    @contextmanager
    def profile_step(self, step_name: str):
        """Context manager for profiling training steps.
        
        Args:
            step_name: Name of the training step
        """
        with self.profiler.profile_memory(step_name):
            yield


def create_memory_optimizer(config: TrainingConfig) -> MemoryOptimizer:
    """Factory function to create MemoryOptimizer.
    
    Args:
        config: Training configuration
        
    Returns:
        MemoryOptimizer instance
    """
    return MemoryOptimizer(config)