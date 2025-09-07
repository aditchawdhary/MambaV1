"""Tests for checkpoint management functionality."""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from mamba_training.training.checkpoint_manager import CheckpointManager, CheckpointMetadata
from mamba_training.config import Config, MambaConfig, TrainingConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


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
        training=TrainingConfig(batch_size=16, learning_rate=1e-4)
    )


@pytest.fixture
def model():
    """Create test model."""
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    """Create test optimizer."""
    return AdamW(model.parameters(), lr=1e-4)


@pytest.fixture
def scheduler(optimizer):
    """Create test scheduler."""
    return CosineAnnealingLR(optimizer, T_max=100)


@pytest.fixture
def checkpoint_manager(temp_dir, config):
    """Create checkpoint manager for testing."""
    return CheckpointManager(temp_dir, config)


class TestCheckpointMetadata:
    """Test CheckpointMetadata class."""
    
    def test_metadata_initialization(self):
        """Test metadata initialization with defaults."""
        metadata = CheckpointMetadata(epoch=5, global_step=1000)
        
        assert metadata.epoch == 5
        assert metadata.global_step == 1000
        assert metadata.validation_errors == []
        assert metadata.is_valid is True
        assert metadata.created_at != ""
    
    def test_metadata_with_custom_values(self):
        """Test metadata with custom values."""
        metadata = CheckpointMetadata(
            epoch=10,
            global_step=2000,
            best_loss=0.5,
            learning_rate=1e-4,
            checkpoint_version="1.0"
        )
        
        assert metadata.epoch == 10
        assert metadata.global_step == 2000
        assert metadata.best_loss == 0.5
        assert metadata.learning_rate == 1e-4
        assert metadata.checkpoint_version == "1.0"


class TestCheckpointManager:
    """Test CheckpointManager class."""
    
    def test_initialization(self, temp_dir, config):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(temp_dir, config)
        
        assert manager.save_dir == temp_dir
        assert manager.config == config
        assert temp_dir.exists()
    
    def test_save_checkpoint_basic(self, checkpoint_manager, model, optimizer, scheduler):
        """Test basic checkpoint saving."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            global_step=100,
            loss=0.5,
            metrics={'accuracy': 0.8}
        )
        
        assert checkpoint_path.exists()
        assert checkpoint_path.name == "checkpoint-100.pt"
        
        # Check metadata file exists
        metadata_path = checkpoint_manager.save_dir / "checkpoint-100.json"
        assert metadata_path.exists()
    
    def test_save_checkpoint_without_scheduler(self, checkpoint_manager, model, optimizer):
        """Test checkpoint saving without scheduler."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            global_step=100,
            loss=0.5
        )
        
        assert checkpoint_path.exists()
        
        # Load and verify checkpoint
        checkpoint_data = torch.load(checkpoint_path)
        assert checkpoint_data['scheduler_state_dict'] is None
    
    def test_save_best_checkpoint(self, checkpoint_manager, model, optimizer, scheduler):
        """Test saving best checkpoint with symlink."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            global_step=100,
            loss=0.3,
            is_best=True
        )
        
        # Check best checkpoint symlink
        best_path = checkpoint_manager.save_dir / "best_checkpoint.pt"
        assert best_path.exists()
        assert best_path.is_symlink()
        assert best_path.resolve().samefile(checkpoint_path)
    
    def test_load_checkpoint_basic(self, checkpoint_manager, model, optimizer, scheduler):
        """Test basic checkpoint loading."""
        # Save checkpoint first
        original_loss = 0.5
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            global_step=100,
            loss=original_loss
        )
        
        # Create new model and optimizer to load into
        new_model = SimpleModel()
        new_optimizer = AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = CosineAnnealingLR(new_optimizer, T_max=100)
        
        # Load checkpoint
        result = checkpoint_manager.load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler
        )
        
        assert result['checkpoint_data']['epoch'] == 1
        assert result['checkpoint_data']['global_step'] == 100
        assert result['checkpoint_data']['loss'] == original_loss
        assert result['metadata'] is not None
    
    def test_load_specific_checkpoint(self, checkpoint_manager, model, optimizer, scheduler):
        """Test loading specific checkpoint by path."""
        # Save multiple checkpoints
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 1, 100, 0.5)
        checkpoint_path_200 = checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 2, 200, 0.4)
        
        # Load specific checkpoint
        result = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path_200,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        assert result['checkpoint_data']['global_step'] == 200
        assert result['checkpoint_data']['loss'] == 0.4
    
    def test_load_nonexistent_checkpoint(self, checkpoint_manager, model):
        """Test loading nonexistent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(
                checkpoint_path="nonexistent.pt",
                model=model
            )
    
    def test_list_checkpoints(self, checkpoint_manager, model, optimizer, scheduler):
        """Test listing available checkpoints."""
        # Save multiple checkpoints
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 1, 100, 0.5)
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 2, 200, 0.4)
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 3, 300, 0.3)
        
        checkpoints = checkpoint_manager.list_checkpoints()
        
        assert len(checkpoints) == 3
        assert checkpoints[0]['global_step'] == 100
        assert checkpoints[1]['global_step'] == 200
        assert checkpoints[2]['global_step'] == 300
        
        # Check that all checkpoints have required fields
        for checkpoint in checkpoints:
            assert 'path' in checkpoint
            assert 'global_step' in checkpoint
            assert 'file_size' in checkpoint
            assert 'created_at' in checkpoint
    
    def test_get_best_checkpoint(self, checkpoint_manager, model, optimizer, scheduler):
        """Test getting best checkpoint."""
        # Initially no best checkpoint
        assert checkpoint_manager.get_best_checkpoint() is None
        
        # Save regular checkpoint
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 1, 100, 0.5)
        assert checkpoint_manager.get_best_checkpoint() is None
        
        # Save best checkpoint
        best_path = checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, 2, 200, 0.3, is_best=True
        )
        
        retrieved_best = checkpoint_manager.get_best_checkpoint()
        assert retrieved_best.samefile(best_path)
    
    def test_validate_checkpoint_valid(self, checkpoint_manager, model, optimizer, scheduler):
        """Test validating a valid checkpoint."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, 1, 100, 0.5
        )
        
        metadata = checkpoint_manager.validate_checkpoint(checkpoint_path)
        
        assert metadata.is_valid is True
        assert len(metadata.validation_errors) == 0
        assert metadata.epoch == 1
        assert metadata.global_step == 100
    
    def test_validate_checkpoint_missing_file(self, checkpoint_manager):
        """Test validating nonexistent checkpoint."""
        metadata = checkpoint_manager.validate_checkpoint("nonexistent.pt")
        
        assert metadata.is_valid is False
        assert "does not exist" in metadata.validation_errors[0]
    
    def test_validate_checkpoint_corrupted(self, checkpoint_manager, temp_dir):
        """Test validating corrupted checkpoint."""
        # Create corrupted checkpoint file
        corrupted_path = temp_dir / "corrupted.pt"
        with open(corrupted_path, 'w') as f:
            f.write("not a valid checkpoint")
        
        metadata = checkpoint_manager.validate_checkpoint(corrupted_path)
        
        assert metadata.is_valid is False
        assert len(metadata.validation_errors) > 0
    
    def test_cleanup_checkpoints(self, checkpoint_manager, model, optimizer, scheduler):
        """Test checkpoint cleanup functionality."""
        # Set max checkpoints to 2
        checkpoint_manager.max_checkpoints = 2
        
        # Save 4 checkpoints
        paths = []
        for i in range(4):
            path = checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, i+1, (i+1)*100, 0.5-i*0.1
            )
            paths.append(path)
        
        # Check that only 2 most recent checkpoints remain
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 2
        assert checkpoints[0]['global_step'] == 300
        assert checkpoints[1]['global_step'] == 400
        
        # Check that old checkpoints were removed
        assert not paths[0].exists()  # checkpoint-100.pt
        assert not paths[1].exists()  # checkpoint-200.pt
    
    def test_cleanup_preserves_best(self, checkpoint_manager, model, optimizer, scheduler):
        """Test that cleanup preserves best checkpoint."""
        checkpoint_manager.max_checkpoints = 2
        
        # Save checkpoints, mark second one as best
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 1, 100, 0.5)
        best_path = checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, 2, 200, 0.3, is_best=True
        )
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 3, 300, 0.4)
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 4, 400, 0.6)
        
        # Best checkpoint should still exist even if it's not in the most recent 2
        assert best_path.exists()
        
        # Verify best checkpoint is still accessible
        retrieved_best = checkpoint_manager.get_best_checkpoint()
        assert retrieved_best.samefile(best_path)
    
    def test_atomic_save_failure_cleanup(self, checkpoint_manager, model, optimizer, scheduler):
        """Test that failed saves clean up temporary files."""
        with patch('torch.save', side_effect=RuntimeError("Save failed")):
            with pytest.raises(RuntimeError):
                checkpoint_manager.save_checkpoint(
                    model, optimizer, scheduler, 1, 100, 0.5
                )
        
        # Check no temporary files left behind
        temp_files = list(checkpoint_manager.save_dir.glob(".*tmp*"))
        assert len(temp_files) == 0
    
    def test_metadata_persistence(self, checkpoint_manager, model, optimizer, scheduler):
        """Test that metadata is correctly saved and loaded."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            global_step=1000,
            loss=0.25,
            metrics={'accuracy': 0.95, 'f1': 0.88}
        )
        
        # Load checkpoint and verify metadata
        result = checkpoint_manager.load_checkpoint(checkpoint_path)
        metadata = result['metadata']
        
        assert metadata.epoch == 5
        assert metadata.global_step == 1000
        assert metadata.model_config is not None
        assert metadata.training_config is not None
        assert metadata.model_hash != ""
        assert metadata.checkpoint_version == "1.0"
    
    def test_symlink_updates(self, checkpoint_manager, model, optimizer, scheduler):
        """Test that symlinks are correctly updated."""
        # Save first checkpoint
        path1 = checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 1, 100, 0.5)
        
        latest_link = checkpoint_manager.save_dir / "latest_checkpoint.pt"
        assert latest_link.exists()
        assert latest_link.resolve().samefile(path1)
        
        # Save second checkpoint
        path2 = checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 2, 200, 0.4)
        
        # Latest should now point to second checkpoint
        assert latest_link.resolve().samefile(path2)
        
        # Save best checkpoint
        best_path = checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, 3, 300, 0.2, is_best=True
        )
        
        best_link = checkpoint_manager.save_dir / "best_checkpoint.pt"
        assert best_link.exists()
        assert best_link.resolve().samefile(best_path)
    
    def test_backward_compatibility_validation(self, checkpoint_manager, model, optimizer, scheduler):
        """Test backward compatibility validation."""
        # Save checkpoint with current version
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, 1, 100, 0.5
        )
        
        # Manually modify metadata to simulate old version
        metadata_path = checkpoint_manager.save_dir / "checkpoint-100.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        metadata_dict['checkpoint_version'] = "0.9"  # Old version
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f)
        
        # Validation should fail for unsupported version
        with pytest.raises(RuntimeError, match="Unsupported checkpoint version"):
            checkpoint_manager.load_checkpoint(checkpoint_path, model=model)
    
    def test_model_hash_computation(self, checkpoint_manager):
        """Test model hash computation for validation."""
        model1 = SimpleModel(10, 20, 5)
        model2 = SimpleModel(10, 20, 5)  # Same architecture
        model3 = SimpleModel(10, 30, 5)  # Different architecture
        
        hash1 = checkpoint_manager._compute_model_hash(model1)
        hash2 = checkpoint_manager._compute_model_hash(model2)
        hash3 = checkpoint_manager._compute_model_hash(model3)
        
        # Same architecture should produce same hash
        assert hash1 == hash2
        
        # Different architecture should produce different hash
        assert hash1 != hash3
        
        # Hash should be consistent
        assert checkpoint_manager._compute_model_hash(model1) == hash1


class TestCheckpointRecovery:
    """Test checkpoint recovery scenarios."""
    
    def test_recovery_from_latest(self, checkpoint_manager, model, optimizer, scheduler):
        """Test recovery from latest checkpoint."""
        # Save multiple checkpoints
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 1, 100, 0.5)
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 2, 200, 0.4)
        latest_path = checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 3, 300, 0.3)
        
        # Load latest checkpoint (without specifying path)
        result = checkpoint_manager.load_checkpoint(model=model, optimizer=optimizer)
        
        assert result['checkpoint_data']['global_step'] == 300
        assert result['checkpoint_path'].samefile(latest_path)
    
    def test_recovery_with_missing_metadata(self, checkpoint_manager, model, optimizer, scheduler):
        """Test recovery when metadata file is missing."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, 1, 100, 0.5
        )
        
        # Remove metadata file
        metadata_path = checkpoint_manager.save_dir / "checkpoint-100.json"
        metadata_path.unlink()
        
        # Should still be able to load checkpoint
        result = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer
        )
        
        assert result['checkpoint_data']['global_step'] == 100
        assert result['metadata'] is None
    
    def test_partial_state_loading(self, checkpoint_manager, model, optimizer, scheduler):
        """Test loading checkpoint with only some components."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, 1, 100, 0.5
        )
        
        # Load only model state
        new_model = SimpleModel()
        result = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model
        )
        
        assert result['checkpoint_data']['global_step'] == 100
        # Optimizer and scheduler states should not be loaded
    
    def test_strict_loading_disabled(self, checkpoint_manager, optimizer, scheduler):
        """Test non-strict loading for model architecture changes."""
        # Create model with different architecture
        original_model = SimpleModel(10, 20, 5)
        checkpoint_path = checkpoint_manager.save_checkpoint(
            original_model, optimizer, scheduler, 1, 100, 0.5
        )
        
        # Create model with extra parameters
        class ExtendedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 5)
                self.extra_layer = nn.Linear(5, 3)  # Extra layer
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.linear2(x)
                return self.extra_layer(x)
        
        extended_model = ExtendedModel()
        
        # Should load successfully with strict=False
        result = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=extended_model,
            strict=False
        )
        
        assert result['checkpoint_data']['global_step'] == 100


if __name__ == "__main__":
    pytest.main([__file__])