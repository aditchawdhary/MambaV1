"""Training monitoring and logging utilities for Mamba training pipeline."""

import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime
import math

import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

from ..config import Config


@dataclass
class MetricValue:
    """Container for a single metric value with metadata."""
    
    value: float
    timestamp: float
    step: int
    epoch: Optional[int] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    # Loss metrics
    loss: float = 0.0
    perplexity: float = 0.0
    
    # Performance metrics
    throughput_tokens_per_sec: float = 0.0
    throughput_samples_per_sec: float = 0.0
    
    # Training progress
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    
    # Memory metrics
    memory_used_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    
    # Timing metrics
    batch_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    
    # Additional custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        result = asdict(self)
        result.update(result.pop('custom_metrics', {}))
        return result


class MetricsTracker:
    """Tracks and aggregates training metrics over time."""
    
    def __init__(self, window_size: int = 100):
        """Initialize metrics tracker.
        
        Args:
            window_size: Size of rolling window for averaging metrics
        """
        self.window_size = window_size
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.global_step = 0
        self.epoch = 0
        
        # Timing state
        self._batch_start_time = None
        self._forward_start_time = None
        self._backward_start_time = None
        
    def start_batch(self) -> None:
        """Mark the start of a training batch."""
        self._batch_start_time = time.time()
    
    def start_forward(self) -> None:
        """Mark the start of forward pass."""
        self._forward_start_time = time.time()
    
    def end_forward(self) -> None:
        """Mark the end of forward pass."""
        if self._forward_start_time is not None:
            forward_time = time.time() - self._forward_start_time
            self.add_metric('forward_time', forward_time)
            self._forward_start_time = None
    
    def start_backward(self) -> None:
        """Mark the start of backward pass."""
        self._backward_start_time = time.time()
    
    def end_backward(self) -> None:
        """Mark the end of backward pass."""
        if self._backward_start_time is not None:
            backward_time = time.time() - self._backward_start_time
            self.add_metric('backward_time', backward_time)
            self._backward_start_time = None
    
    def end_batch(self, batch_size: int, sequence_length: int) -> None:
        """Mark the end of a training batch and compute throughput.
        
        Args:
            batch_size: Number of samples in the batch
            sequence_length: Length of sequences in the batch
        """
        if self._batch_start_time is not None:
            batch_time = time.time() - self._batch_start_time
            self.add_metric('batch_time', batch_time)
            
            # Compute throughput metrics
            if batch_time > 0:
                samples_per_sec = batch_size / batch_time
                tokens_per_sec = (batch_size * sequence_length) / batch_time
                
                self.add_metric('throughput_samples_per_sec', samples_per_sec)
                self.add_metric('throughput_tokens_per_sec', tokens_per_sec)
            
            self._batch_start_time = None
    
    def add_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Add a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Global step (uses internal counter if None)
        """
        if step is None:
            step = self.global_step
        
        metric_value = MetricValue(
            value=value,
            timestamp=time.time(),
            step=step,
            epoch=self.epoch
        )
        
        self.metrics_history[name].append(metric_value)
    
    def update_loss(self, loss: float) -> None:
        """Update loss and compute perplexity.
        
        Args:
            loss: Loss value
        """
        self.add_metric('loss', loss)
        
        # Compute perplexity (exp(loss))
        try:
            perplexity = math.exp(loss)
            # Cap perplexity to avoid overflow
            perplexity = min(perplexity, 1e6)
            self.add_metric('perplexity', perplexity)
        except (OverflowError, ValueError):
            self.add_metric('perplexity', float('inf'))
    
    def update_learning_rate(self, lr: float) -> None:
        """Update learning rate metric.
        
        Args:
            lr: Learning rate value
        """
        self.add_metric('learning_rate', lr)
    
    def update_gradient_norm(self, grad_norm: float) -> None:
        """Update gradient norm metric.
        
        Args:
            grad_norm: Gradient norm value
        """
        self.add_metric('gradient_norm', grad_norm)
    
    def update_memory_usage(self) -> None:
        """Update GPU memory usage metrics."""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            
            self.add_metric('memory_used_gb', memory_used)
            self.add_metric('memory_reserved_gb', memory_reserved)
    
    def get_current_metrics(self) -> TrainingMetrics:
        """Get current training metrics.
        
        Returns:
            TrainingMetrics: Current metrics snapshot
        """
        metrics = TrainingMetrics()
        
        # Get latest values for each metric
        for metric_name, history in self.metrics_history.items():
            if history:
                latest_value = history[-1].value
                
                if hasattr(metrics, metric_name):
                    setattr(metrics, metric_name, latest_value)
                else:
                    metrics.custom_metrics[metric_name] = latest_value
        
        return metrics
    
    def get_averaged_metrics(self, window_size: Optional[int] = None) -> TrainingMetrics:
        """Get averaged metrics over a window.
        
        Args:
            window_size: Size of averaging window (uses default if None)
            
        Returns:
            TrainingMetrics: Averaged metrics
        """
        if window_size is None:
            window_size = self.window_size
        
        metrics = TrainingMetrics()
        
        # Compute averages for each metric
        for metric_name, history in self.metrics_history.items():
            if history:
                recent_values = list(history)[-window_size:]
                avg_value = sum(mv.value for mv in recent_values) / len(recent_values)
                
                if hasattr(metrics, metric_name):
                    setattr(metrics, metric_name, avg_value)
                else:
                    metrics.custom_metrics[metric_name] = avg_value
        
        return metrics
    
    def step(self) -> None:
        """Increment global step counter."""
        self.global_step += 1
    
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch
    
    def reset(self) -> None:
        """Reset all metrics history."""
        self.metrics_history.clear()
        self.global_step = 0
        self.epoch = 0


class TrainingLogger:
    """Configurable logging utility for training progress."""
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str = "mamba_training",
        log_level: str = "INFO",
        console_log: bool = True,
        file_log: bool = True,
        tensorboard_log: bool = True
    ):
        """Initialize training logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            console_log: Whether to log to console
            file_log: Whether to log to file
            tensorboard_log: Whether to log to TensorBoard
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.console_log = console_log
        self.file_log = file_log
        self.tensorboard_log = tensorboard_log
        
        # Setup Python logger
        self.logger = logging.getLogger(f"mamba_training.{experiment_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup console logging
        if console_log:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Setup file logging
        if file_log:
            log_file = self.log_dir / f"{experiment_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Setup TensorBoard logging
        self.tb_writer = None
        if tensorboard_log and TENSORBOARD_AVAILABLE:
            tb_dir = self.log_dir / "tensorboard" / experiment_name
            self.tb_writer = SummaryWriter(tb_dir)
        elif tensorboard_log and not TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard not available, skipping TensorBoard logging")
        
        # Metrics file for JSON logging
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
    
    def log_metrics(
        self,
        metrics: TrainingMetrics,
        step: int,
        epoch: Optional[int] = None,
        prefix: str = "train"
    ) -> None:
        """Log training metrics.
        
        Args:
            metrics: Metrics to log
            step: Global step number
            epoch: Epoch number
            prefix: Prefix for metric names (e.g., 'train', 'val')
        """
        metrics_dict = metrics.to_dict()
        
        # Log to console/file
        metric_strs = []
        for name, value in metrics_dict.items():
            if isinstance(value, float):
                if name in ['loss', 'perplexity', 'learning_rate']:
                    metric_strs.append(f"{name}={value:.6f}")
                elif 'time' in name:
                    metric_strs.append(f"{name}={value:.3f}s")
                elif 'throughput' in name:
                    metric_strs.append(f"{name}={value:.1f}")
                elif 'memory' in name:
                    metric_strs.append(f"{name}={value:.2f}GB")
                else:
                    metric_strs.append(f"{name}={value:.4f}")
        
        epoch_str = f"Epoch {epoch}, " if epoch is not None else ""
        self.logger.info(f"{epoch_str}Step {step}: {', '.join(metric_strs)}")
        
        # Log to TensorBoard
        if self.tb_writer:
            for name, value in metrics_dict.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    self.tb_writer.add_scalar(f"{prefix}/{name}", value, step)
        
        # Log to JSON file
        if self.file_log:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'step': step,
                'epoch': epoch,
                'prefix': prefix,
                'metrics': metrics_dict
            }
            
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def log_hyperparameters(self, config: Config) -> None:
        """Log hyperparameters.
        
        Args:
            config: Training configuration
        """
        if self.tb_writer:
            # Convert config to flat dictionary for TensorBoard
            hparams = {}
            
            # Model hyperparameters
            hparams.update({
                f"model/{k}": v for k, v in asdict(config.model).items()
                if isinstance(v, (int, float, str, bool))
            })
            
            # Training hyperparameters
            hparams.update({
                f"training/{k}": v for k, v in asdict(config.training).items()
                if isinstance(v, (int, float, str, bool))
            })
            
            # Data hyperparameters
            hparams.update({
                f"data/{k}": v for k, v in asdict(config.data).items()
                if isinstance(v, (int, float, str, bool))
            })
            
            self.tb_writer.add_hparams(hparams, {})
        
        # Log to file
        self.logger.info(f"Model config: {asdict(config.model)}")
        self.logger.info(f"Training config: {asdict(config.training)}")
        self.logger.info(f"Data config: {asdict(config.data)}")
    
    def log_model_info(self, model: nn.Module) -> None:
        """Log model architecture information.
        
        Args:
            model: Model to analyze
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log model architecture
        self.logger.debug(f"Model architecture:\n{model}")
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_text("model/architecture", str(model))
            self.tb_writer.add_scalar("model/total_parameters", total_params)
            self.tb_writer.add_scalar("model/trainable_parameters", trainable_params)
    
    def log_training_start(self, total_steps: int, total_epochs: int) -> None:
        """Log training start information.
        
        Args:
            total_steps: Total number of training steps
            total_epochs: Total number of epochs
        """
        self.logger.info(f"Starting training: {total_epochs} epochs, {total_steps} steps")
        self.logger.info(f"Logs will be saved to: {self.log_dir}")
    
    def log_training_end(self, total_time: float, final_metrics: TrainingMetrics) -> None:
        """Log training completion information.
        
        Args:
            total_time: Total training time in seconds
            final_metrics: Final training metrics
        """
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        
        self.logger.info(f"Training completed in {hours:.0f}h {minutes:.0f}m {seconds:.1f}s")
        self.logger.info(f"Final metrics: {final_metrics.to_dict()}")
    
    def log_checkpoint_saved(self, checkpoint_path: Path, metrics: TrainingMetrics) -> None:
        """Log checkpoint saving.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            metrics: Current metrics
        """
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.logger.debug(f"Checkpoint metrics: {metrics.to_dict()}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log training error.
        
        Args:
            error: Exception that occurred
            context: Additional context about the error
        """
        error_msg = f"Training error"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(error)}"
        
        self.logger.error(error_msg, exc_info=True)
    
    def close(self) -> None:
        """Close logger and cleanup resources."""
        if self.tb_writer:
            self.tb_writer.close()
        
        # Close file handlers
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()


class ProgressReporter:
    """Reports training progress with visualization."""
    
    def __init__(
        self,
        total_steps: int,
        report_interval: int = 100,
        logger: Optional[TrainingLogger] = None
    ):
        """Initialize progress reporter.
        
        Args:
            total_steps: Total number of training steps
            report_interval: Steps between progress reports
            logger: Optional logger for output
        """
        self.total_steps = total_steps
        self.report_interval = report_interval
        self.logger = logger
        
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.last_report_step = 0
    
    def report_progress(
        self,
        step: int,
        metrics: TrainingMetrics,
        epoch: Optional[int] = None
    ) -> None:
        """Report training progress.
        
        Args:
            step: Current step
            metrics: Current metrics
            epoch: Current epoch
        """
        if step % self.report_interval != 0 and step != self.total_steps:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Compute progress statistics
        progress_pct = (step / self.total_steps) * 100
        
        # Estimate remaining time
        if step > 0:
            time_per_step = elapsed_time / step
            remaining_steps = self.total_steps - step
            eta_seconds = remaining_steps * time_per_step
            
            eta_hours = eta_seconds // 3600
            eta_minutes = (eta_seconds % 3600) // 60
            eta_str = f"{eta_hours:.0f}h {eta_minutes:.0f}m"
        else:
            eta_str = "unknown"
        
        # Compute recent throughput
        steps_since_last = step - self.last_report_step
        time_since_last = current_time - self.last_report_time
        
        if time_since_last > 0 and steps_since_last > 0:
            recent_steps_per_sec = steps_since_last / time_since_last
        else:
            recent_steps_per_sec = 0
        
        # Format progress message
        epoch_str = f"Epoch {epoch}, " if epoch is not None else ""
        progress_msg = (
            f"{epoch_str}Step {step}/{self.total_steps} ({progress_pct:.1f}%) | "
            f"Loss: {metrics.loss:.4f} | "
            f"PPL: {metrics.perplexity:.2f} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Throughput: {metrics.throughput_tokens_per_sec:.0f} tok/s | "
            f"ETA: {eta_str}"
        )
        
        if self.logger:
            self.logger.logger.info(progress_msg)
        else:
            print(progress_msg)
        
        # Update tracking variables
        self.last_report_time = current_time
        self.last_report_step = step
    
    def create_progress_bar(self, step: int, width: int = 50) -> str:
        """Create ASCII progress bar.
        
        Args:
            step: Current step
            width: Width of progress bar
            
        Returns:
            str: ASCII progress bar
        """
        progress = step / self.total_steps
        filled_width = int(width * progress)
        
        bar = "█" * filled_width + "░" * (width - filled_width)
        return f"[{bar}] {progress*100:.1f}%"


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        float: Perplexity value
    """
    try:
        perplexity = math.exp(loss)
        # Cap perplexity to avoid overflow
        return min(perplexity, 1e6)
    except (OverflowError, ValueError):
        return float('inf')


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute gradient norm for model parameters.
    
    Args:
        model: Model to compute gradient norm for
        
    Returns:
        float: Gradient norm
    """
    total_norm = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count == 0:
        return 0.0
    
    return total_norm ** 0.5


def create_training_monitor(
    log_dir: Union[str, Path],
    config: Config,
    total_steps: int,
    experiment_name: Optional[str] = None
) -> tuple[MetricsTracker, TrainingLogger, ProgressReporter]:
    """Create complete training monitoring setup.
    
    Args:
        log_dir: Directory for logs
        config: Training configuration
        total_steps: Total training steps
        experiment_name: Name of experiment
        
    Returns:
        tuple: (metrics_tracker, logger, progress_reporter)
    """
    if experiment_name is None:
        experiment_name = config.experiment_name
    
    # Create metrics tracker
    metrics_tracker = MetricsTracker(window_size=100)
    
    # Create logger
    logger = TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        log_level=config.log_level,
        console_log=True,
        file_log=True,
        tensorboard_log=True
    )
    
    # Create progress reporter
    progress_reporter = ProgressReporter(
        total_steps=total_steps,
        report_interval=config.training.save_steps // 10,  # Report 10x more frequently than saving
        logger=logger
    )
    
    return metrics_tracker, logger, progress_reporter