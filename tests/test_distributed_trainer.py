"""Tests for distributed training engine."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mamba_training.training.distributed_trainer import (
    DistributedTrainer,
    create_distributed_trainer,
    setup_distributed_sampler
)
from mamba_training.config import Config, TrainingConfig, MambaConfig, DataConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, vocab_size: int = 100, d_model: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        x = x.mean(dim=1)  # Simple pooling
        logits = self.linear(x)
        
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits}


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        model=MambaConfig(d_model=64, vocab_size=100, n_layers=2),
        training=TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-3,
            num_epochs=2,
            warmup_steps=5,
            save_steps=10,
            eval_steps=5,
            use_mixed_precision=False
        ),
        data=DataConfig(max_seq_length=32),
        experiment_name="test_experiment"
    )


@pytest.fixture
def dummy_dataset():
    """Create dummy dataset for testing."""
    # Create random input_ids and labels
    input_ids = torch.randint(0, 100, (50, 32))  # 50 samples, seq_len 32
    labels = torch.randint(0, 100, (50,))  # 50 labels
    
    return TensorDataset(input_ids, labels)


@pytest.fixture
def train_dataloader(dummy_dataset):
    """Create training dataloader."""
    return DataLoader(dummy_dataset, batch_size=4, shuffle=True)


@pytest.fixture
def val_dataloader(dummy_dataset):
    """Create validation dataloader."""
    return DataLoader(dummy_dataset, batch_size=4, shuffle=False)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestDistributedTrainer:
    """Test DistributedTrainer functionality."""
    
    def test_initialization_single_gpu(self, simple_model, config, train_dataloader, temp_output_dir):
        """Test trainer initialization in single GPU mode."""
        config.output_dir = temp_output_dir
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = DistributedTrainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader
            )
        
        assert trainer.world_size == 1
        assert trainer.rank == 0
        assert trainer.local_rank == 0
        assert not trainer.is_distributed
        assert trainer.model is not None
        assert trainer.optimizer_manager is not None
        assert trainer.total_steps > 0
    
    @patch('mamba_training.training.distributed_trainer.DDP')
    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.is_initialized', return_value=False)
    def test_initialization_distributed(self, mock_is_init, mock_init_pg, mock_ddp, simple_model, config, train_dataloader, temp_output_dir):
        """Test trainer initialization in distributed mode."""
        config.output_dir = temp_output_dir
        
        # Mock DDP to return the original model
        mock_ddp.return_value = simple_model
        
        with patch.dict('os.environ', {'WORLD_SIZE': '2', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = DistributedTrainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader
            )
        
        assert trainer.world_size == 2
        assert trainer.rank == 0
        assert trainer.is_distributed
        mock_init_pg.assert_called_once()
    
    def test_model_wrapping_single_gpu(self, simple_model, config, train_dataloader, temp_output_dir):
        """Test model wrapping for single GPU."""
        config.output_dir = temp_output_dir
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = DistributedTrainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader
            )
        
        # Model should not be wrapped in DDP/FSDP for single GPU
        assert not hasattr(trainer.model, 'module')
    
    def test_total_steps_calculation(self, simple_model, config, train_dataloader, temp_output_dir):
        """Test total steps calculation."""
        config.output_dir = temp_output_dir
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = DistributedTrainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader
            )
        
        expected_steps_per_epoch = len(train_dataloader) // config.training.gradient_accumulation_steps
        expected_total_steps = expected_steps_per_epoch * config.training.num_epochs
        
        assert trainer.total_steps == expected_total_steps
    
    def test_train_step(self, simple_model, config, train_dataloader, temp_output_dir):
        """Test single training step."""
        config.output_dir = temp_output_dir
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = DistributedTrainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader
            )
        
        # Get a batch from dataloader
        batch_data = next(iter(train_dataloader))
        batch = {
            'input_ids': batch_data[0],
            'labels': batch_data[1]
        }
        
        # Perform training step
        metrics = trainer.train_step(batch)
        
        assert 'loss' in metrics
        assert 'learning_rate' in metrics
        assert 'grad_norm' in metrics
        assert 'global_step' in metrics
        assert isinstance(metrics['loss'], float)
        assert metrics['loss'] > 0
    
    def test_validation(self, simple_model, config, train_dataloader, val_dataloader, temp_output_dir):
        """Test validation loop."""
        config.output_dir = temp_output_dir
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = DistributedTrainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader
            )
        
        val_metrics = trainer.validate()
        
        assert 'loss' in val_metrics
        assert 'perplexity' in val_metrics
        assert 'num_batches' in val_metrics
        assert 'is_best' in val_metrics
        assert isinstance(val_metrics['loss'], float)
        assert val_metrics['loss'] > 0
        assert val_metrics['perplexity'] > 0
    
    def test_train_epoch(self, simple_model, config, train_dataloader, temp_output_dir):
        """Test training for one epoch."""
        config.output_dir = temp_output_dir
        config.training.gradient_accumulation_steps = 1  # Simplify for testing
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = DistributedTrainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader
            )
        
        epoch_metrics = trainer.train_epoch()
        
        assert 'loss' in epoch_metrics
        assert 'tokens_per_second' in epoch_metrics
        assert 'epoch_time' in epoch_metrics
        assert 'num_batches' in epoch_metrics
        assert isinstance(epoch_metrics['loss'], float)
        assert epoch_metrics['loss'] > 0
        assert epoch_metrics['num_batches'] > 0
    
    def test_checkpoint_save_load(self, simple_model, config, train_dataloader, temp_output_dir):
        """Test checkpoint saving and loading."""
        config.output_dir = temp_output_dir
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = DistributedTrainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader
            )
        
        # Train for a few steps
        trainer.global_step = 10
        trainer.current_epoch = 1
        
        # Save checkpoint
        metrics = {'loss': 1.5, 'epoch': 1}
        trainer.save_checkpoint(metrics)
        
        # Check checkpoint file exists
        checkpoint_path = Path(temp_output_dir) / 'checkpoint_epoch_1.pt'
        assert checkpoint_path.exists()
        
        # Create new trainer and load checkpoint
        new_trainer = DistributedTrainer(
            model=SimpleModel(),
            config=config,
            train_dataloader=train_dataloader,
            resume_from_checkpoint=str(checkpoint_path)
        )
        
        assert new_trainer.global_step == 10
        assert new_trainer.current_epoch == 1
    
    def test_full_training_loop(self, simple_model, config, train_dataloader, val_dataloader, temp_output_dir):
        """Test complete training loop."""
        config.output_dir = temp_output_dir
        config.training.num_epochs = 1  # Short training for test
        config.training.gradient_accumulation_steps = 1
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = DistributedTrainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader
            )
        
        results = trainer.train()
        
        assert 'total_time' in results
        assert 'final_metrics' in results
        assert 'training_history' in results
        assert results['total_time'] > 0
        assert len(results['training_history']) == config.training.num_epochs
    
    def test_move_batch_to_device(self, simple_model, config, train_dataloader, temp_output_dir):
        """Test batch device movement."""
        config.output_dir = temp_output_dir
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = DistributedTrainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader
            )
        
        batch = {
            'input_ids': torch.randint(0, 100, (4, 32)),
            'labels': torch.randint(0, 100, (4,)),
            'metadata': 'some_string'  # Non-tensor value
        }
        
        moved_batch = trainer._move_batch_to_device(batch)
        
        assert moved_batch['input_ids'].device == trainer.device
        assert moved_batch['labels'].device == trainer.device
        assert moved_batch['metadata'] == 'some_string'  # String unchanged


class TestDistributedSampler:
    """Test distributed sampler setup."""
    
    def test_single_gpu_sampler(self, dummy_dataset):
        """Test sampler setup for single GPU."""
        with patch.dict('os.environ', {'WORLD_SIZE': '1'}):
            dataloader, sampler = setup_distributed_sampler(
                dummy_dataset, batch_size=4, shuffle=True
            )
        
        assert sampler is None  # No distributed sampler needed
        assert dataloader.batch_size == 4
        # DataLoader doesn't expose shuffle attribute directly, but we can check sampler is None
    
    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_world_size', return_value=2)
    @patch('torch.distributed.get_rank', return_value=0)
    def test_distributed_sampler(self, mock_rank, mock_world_size, mock_is_init, mock_init_pg, dummy_dataset):
        """Test sampler setup for distributed training."""
        with patch.dict('os.environ', {'WORLD_SIZE': '2', 'RANK': '0'}):
            dataloader, sampler = setup_distributed_sampler(
                dummy_dataset, batch_size=4, shuffle=True
            )
        
        assert sampler is not None
        # DataLoader shuffle should be False when using distributed sampler


class TestFactoryFunction:
    """Test factory function."""
    
    def test_create_distributed_trainer(self, simple_model, config, train_dataloader, temp_output_dir):
        """Test factory function."""
        config.output_dir = temp_output_dir
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = create_distributed_trainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader
            )
        
        assert isinstance(trainer, DistributedTrainer)
        assert trainer.model == simple_model
        assert trainer.config == config


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_checkpoint(self, simple_model, config, train_dataloader, temp_output_dir):
        """Test loading non-existent checkpoint."""
        config.output_dir = temp_output_dir
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            with pytest.raises(FileNotFoundError):
                DistributedTrainer(
                    model=simple_model,
                    config=config,
                    train_dataloader=train_dataloader,
                    resume_from_checkpoint="non_existent_checkpoint.pt"
                )
    
    def test_validation_without_dataloader(self, simple_model, config, train_dataloader, temp_output_dir):
        """Test validation when no validation dataloader is provided."""
        config.output_dir = temp_output_dir
        
        with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            trainer = DistributedTrainer(
                model=simple_model,
                config=config,
                train_dataloader=train_dataloader,
                val_dataloader=None
            )
        
        val_metrics = trainer.validate()
        assert val_metrics == {}


@pytest.mark.parametrize("use_mixed_precision", [True, False])
def test_mixed_precision_training(simple_model, config, train_dataloader, temp_output_dir, use_mixed_precision):
    """Test training with and without mixed precision."""
    config.output_dir = temp_output_dir
    config.training.use_mixed_precision = use_mixed_precision
    
    with patch.dict('os.environ', {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
        trainer = DistributedTrainer(
            model=simple_model,
            config=config,
            train_dataloader=train_dataloader
        )
    
    # Get a batch and perform training step
    batch_data = next(iter(train_dataloader))
    batch = {
        'input_ids': batch_data[0],
        'labels': batch_data[1]
    }
    
    metrics = trainer.train_step(batch)
    
    assert 'loss' in metrics
    assert isinstance(metrics['loss'], float)
    assert metrics['loss'] > 0


if __name__ == "__main__":
    pytest.main([__file__])