"""Checkpoint management for Mamba training pipeline."""

import os
import json
import time
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..config import Config


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files."""
    
    # Training state
    epoch: int
    global_step: int
    best_loss: Optional[float] = None
    learning_rate: float = 0.0
    
    # Model information
    model_config: Dict[str, Any] = None
    model_hash: str = ""
    
    # Training configuration
    training_config: Dict[str, Any] = None
    
    # Checkpoint metadata
    checkpoint_version: str = "1.0"
    created_at: str = ""
    file_size: int = 0
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.validation_errors is None:
            self.validation_errors = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class CheckpointManager:
    """Manages model checkpoints with atomic writing and validation."""
    
    def __init__(self, save_dir: Union[str, Path], config: Optional[Config] = None):
        """Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            config: Training configuration for metadata
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Checkpoint file patterns
        self.checkpoint_pattern = "checkpoint-{step}.pt"
        self.metadata_pattern = "checkpoint-{step}.json"
        self.best_checkpoint_name = "best_checkpoint.pt"
        self.latest_checkpoint_name = "latest_checkpoint.pt"
        
        # Validation settings
        self.max_checkpoints = 5  # Keep only last N checkpoints
        self.validate_on_load = True
        
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        global_step: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> Path:
        """Save model checkpoint with atomic writing.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler state
            epoch: Current epoch number
            global_step: Global training step
            loss: Current loss value
            metrics: Additional metrics to save
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path: Path to saved checkpoint file
            
        Raises:
            RuntimeError: If checkpoint saving fails
        """
        if metrics is None:
            metrics = {}
            
        # Create checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'global_step': global_step,
            'loss': loss,
            'metrics': metrics,
            'config': asdict(self.config) if self.config else None,
        }
        
        # Generate checkpoint filename
        checkpoint_filename = self.checkpoint_pattern.format(step=global_step)
        checkpoint_path = self.save_dir / checkpoint_filename
        
        # Create metadata
        metadata = CheckpointMetadata(
            epoch=epoch,
            global_step=global_step,
            best_loss=loss if is_best else None,
            learning_rate=optimizer.param_groups[0]['lr'],
            model_config=asdict(self.config.model) if self.config else None,
            training_config=asdict(self.config.training) if self.config else None,
            model_hash=self._compute_model_hash(model),
        )
        
        try:
            # Atomic checkpoint writing using temporary file
            temp_checkpoint_path = self._atomic_save(checkpoint_data, checkpoint_path)
            
            # Update file size in metadata
            metadata.file_size = temp_checkpoint_path.stat().st_size
            
            # Save metadata
            metadata_path = self.save_dir / self.metadata_pattern.format(step=global_step)
            self._save_metadata(metadata, metadata_path)
            
            # Update symlinks for latest and best checkpoints
            self._update_symlinks(checkpoint_path, is_best)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return checkpoint_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint: {e}") from e
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """Load checkpoint with validation.
        
        Args:
            checkpoint_path: Path to checkpoint file (None for latest)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            strict: Whether to enforce strict state dict loading
            
        Returns:
            Dict: Checkpoint data and metadata
            
        Raises:
            FileNotFoundError: If checkpoint file not found
            RuntimeError: If checkpoint loading or validation fails
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()
        else:
            checkpoint_path = Path(checkpoint_path)
            
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Load and validate metadata if available
            metadata = self._load_metadata(checkpoint_path)
            if metadata and self.validate_on_load:
                self._validate_checkpoint(checkpoint_data, metadata)
            
            # Load states into provided objects
            if model is not None:
                model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            if scheduler is not None and checkpoint_data.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            # Return checkpoint data with metadata
            result = {
                'checkpoint_data': checkpoint_data,
                'metadata': metadata,
                'checkpoint_path': checkpoint_path,
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}") from e
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata.
        
        Returns:
            List: List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for checkpoint_file in self.save_dir.glob("checkpoint-*.pt"):
            try:
                # Extract step number from filename
                step_str = checkpoint_file.stem.split('-')[1]
                global_step = int(step_str)
                
                # Load metadata if available
                metadata_file = self.save_dir / self.metadata_pattern.format(step=global_step)
                metadata = None
                if metadata_file.exists():
                    metadata = self._load_metadata(checkpoint_file)
                
                checkpoint_info = {
                    'path': checkpoint_file,
                    'global_step': global_step,
                    'file_size': checkpoint_file.stat().st_size,
                    'created_at': datetime.fromtimestamp(checkpoint_file.stat().st_mtime),
                    'metadata': metadata,
                }
                
                checkpoints.append(checkpoint_info)
                
            except (ValueError, IndexError):
                # Skip files that don't match expected pattern
                continue
        
        # Sort by global step
        checkpoints.sort(key=lambda x: x['global_step'])
        return checkpoints
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint.
        
        Returns:
            Optional[Path]: Path to best checkpoint or None if not found
        """
        best_path = self.save_dir / self.best_checkpoint_name
        if best_path.exists() and best_path.is_symlink():
            return best_path.resolve()
        return None
    
    def validate_checkpoint(self, checkpoint_path: Union[str, Path]) -> CheckpointMetadata:
        """Validate checkpoint integrity and compatibility.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            CheckpointMetadata: Validation results
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Initialize validation metadata
        metadata = CheckpointMetadata(
            epoch=0,
            global_step=0,
            is_valid=True,
            validation_errors=[]
        )
        
        try:
            # Check file exists and is readable
            if not checkpoint_path.exists():
                metadata.is_valid = False
                metadata.validation_errors.append("Checkpoint file does not exist")
                return metadata
            
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Validate required fields
            required_fields = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'global_step']
            for field in required_fields:
                if field not in checkpoint_data:
                    metadata.validation_errors.append(f"Missing required field: {field}")
            
            # Load existing metadata if available
            existing_metadata = self._load_metadata(checkpoint_path)
            if existing_metadata:
                # Validate model hash if available
                if existing_metadata.model_hash:
                    # Note: We can't validate model hash without the actual model
                    pass
                
                # Copy metadata fields
                metadata.epoch = existing_metadata.epoch
                metadata.global_step = existing_metadata.global_step
                metadata.model_config = existing_metadata.model_config
                metadata.training_config = existing_metadata.training_config
            
            # Set validation result
            metadata.is_valid = len(metadata.validation_errors) == 0
            
        except Exception as e:
            metadata.is_valid = False
            metadata.validation_errors.append(f"Failed to load checkpoint: {str(e)}")
        
        return metadata
    
    def cleanup_checkpoints(self, keep_best: bool = True, keep_latest: bool = True) -> int:
        """Clean up old checkpoints keeping only specified ones.
        
        Args:
            keep_best: Whether to keep the best checkpoint
            keep_latest: Whether to keep the latest checkpoint
            
        Returns:
            int: Number of checkpoints removed
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.max_checkpoints:
            return 0
        
        # Determine which checkpoints to keep
        keep_paths = set()
        
        if keep_best:
            best_path = self.get_best_checkpoint()
            if best_path:
                keep_paths.add(best_path)
        
        if keep_latest:
            latest_path = self._get_latest_checkpoint()
            if latest_path:
                keep_paths.add(latest_path)
        
        # Keep the most recent checkpoints
        recent_checkpoints = sorted(checkpoints, key=lambda x: x['global_step'])[-self.max_checkpoints:]
        for checkpoint in recent_checkpoints:
            keep_paths.add(checkpoint['path'])
        
        # Remove old checkpoints
        removed_count = 0
        for checkpoint in checkpoints:
            if checkpoint['path'] not in keep_paths:
                try:
                    checkpoint['path'].unlink()
                    # Also remove metadata file
                    metadata_file = self.save_dir / self.metadata_pattern.format(
                        step=checkpoint['global_step']
                    )
                    if metadata_file.exists():
                        metadata_file.unlink()
                    removed_count += 1
                except OSError:
                    pass  # Ignore errors when removing files
        
        return removed_count
    
    def _atomic_save(self, data: Dict[str, Any], target_path: Path) -> Path:
        """Save data atomically using temporary file.
        
        Args:
            data: Data to save
            target_path: Final path for the file
            
        Returns:
            Path: Path to the saved file
        """
        # Create temporary file in same directory
        temp_dir = target_path.parent
        with tempfile.NamedTemporaryFile(
            dir=temp_dir,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Save to temporary file
            torch.save(data, temp_path)
            
            # Atomic move to final location
            shutil.move(str(temp_path), str(target_path))
            
            return target_path
            
        except Exception:
            # Clean up temporary file on error
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def _save_metadata(self, metadata: CheckpointMetadata, metadata_path: Path) -> None:
        """Save checkpoint metadata to JSON file.
        
        Args:
            metadata: Metadata to save
            metadata_path: Path to save metadata
        """
        metadata_dict = asdict(metadata)
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(metadata_dict, temp_file, indent=2)
        
        try:
            shutil.move(str(temp_path), str(metadata_path))
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def _load_metadata(self, checkpoint_path: Path) -> Optional[CheckpointMetadata]:
        """Load checkpoint metadata from JSON file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Optional[CheckpointMetadata]: Loaded metadata or None
        """
        # Extract step number from checkpoint filename
        try:
            step_str = checkpoint_path.stem.split('-')[1]
            global_step = int(step_str)
            metadata_path = self.save_dir / self.metadata_pattern.format(step=global_step)
            
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            return CheckpointMetadata(**metadata_dict)
            
        except (ValueError, IndexError, json.JSONDecodeError, TypeError):
            return None
    
    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint.
        
        Returns:
            Optional[Path]: Path to latest checkpoint or None
        """
        latest_path = self.save_dir / self.latest_checkpoint_name
        if latest_path.exists() and latest_path.is_symlink():
            return latest_path.resolve()
        
        # Fallback: find latest checkpoint by step number
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[-1]['path']
        
        return None
    
    def _update_symlinks(self, checkpoint_path: Path, is_best: bool) -> None:
        """Update symlinks for latest and best checkpoints.
        
        Args:
            checkpoint_path: Path to checkpoint file
            is_best: Whether this is the best checkpoint
        """
        # Update latest checkpoint symlink
        latest_link = self.save_dir / self.latest_checkpoint_name
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_path.name)
        
        # Update best checkpoint symlink if this is the best
        if is_best:
            best_link = self.save_dir / self.best_checkpoint_name
            if best_link.exists():
                best_link.unlink()
            best_link.symlink_to(checkpoint_path.name)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return
        
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Keep the most recent checkpoints
        checkpoints_to_remove = checkpoints[:-self.max_checkpoints]
        
        for checkpoint in checkpoints_to_remove:
            try:
                # Don't remove if it's the best checkpoint
                best_path = self.get_best_checkpoint()
                if best_path and checkpoint['path'].samefile(best_path):
                    continue
                
                checkpoint['path'].unlink()
                
                # Also remove metadata file
                metadata_file = self.save_dir / self.metadata_pattern.format(
                    step=checkpoint['global_step']
                )
                if metadata_file.exists():
                    metadata_file.unlink()
                    
            except OSError:
                pass  # Ignore errors when removing files
    
    def _compute_model_hash(self, model: nn.Module) -> str:
        """Compute hash of model architecture for validation.
        
        Args:
            model: Model to hash
            
        Returns:
            str: SHA256 hash of model structure
        """
        # Create a string representation of model structure
        model_str = str(model)
        
        # Include parameter shapes for more robust validation
        param_shapes = []
        for name, param in model.named_parameters():
            param_shapes.append(f"{name}:{list(param.shape)}")
        
        combined_str = model_str + "|" + "|".join(param_shapes)
        
        return hashlib.sha256(combined_str.encode()).hexdigest()
    
    def _validate_checkpoint(self, checkpoint_data: Dict[str, Any], metadata: CheckpointMetadata) -> None:
        """Validate checkpoint data against metadata.
        
        Args:
            checkpoint_data: Loaded checkpoint data
            metadata: Checkpoint metadata
            
        Raises:
            RuntimeError: If validation fails
        """
        errors = []
        
        # Validate epoch and step consistency
        if checkpoint_data.get('epoch') != metadata.epoch:
            errors.append(f"Epoch mismatch: data={checkpoint_data.get('epoch')}, metadata={metadata.epoch}")
        
        if checkpoint_data.get('global_step') != metadata.global_step:
            errors.append(f"Global step mismatch: data={checkpoint_data.get('global_step')}, metadata={metadata.global_step}")
        
        # Validate checkpoint version compatibility
        if metadata.checkpoint_version != "1.0":
            errors.append(f"Unsupported checkpoint version: {metadata.checkpoint_version}")
        
        if errors:
            raise RuntimeError(f"Checkpoint validation failed: {'; '.join(errors)}")