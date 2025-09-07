"""
Integration and end-to-end tests for Mamba training pipeline.

This module implements comprehensive integration tests for:
- Full training pipeline tests with small datasets
- Distributed training validation across multiple processes
- Checkpoint recovery and model loading tests
- Documentation and usage examples

Requirements: 4.1, 4.2, 5.4
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import tempfile
import shutil
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import subprocess
import sys

from mamba_training.models.mamba_model import MambaModel
from mamba_training.config import MambaConfig, TrainingConfig, DataConfig, Config
from mamba_training.training.distributed_trainer import DistributedTrainer
from mamba_training.training.checkpoint_manager import CheckpointManager
from mamba_training.data.data_loader import MambaDataLoader, create_data_loaders
from mamba_training.data.dataset_processor import ProcessedSample, ProcessedDataset


@dataclass
class IntegrationTestResults:
    """Container for integration test results."""
    test_name: str
    passed: bool
    duration: float
    metrics: Dict[str, Any]
    error_message: str = ""


class MockDataset(torch.utils.data.Dataset):
    """Mock dataset for testing purposes."""
    
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        """Initialize mock dataset.
        
        Args:
            num_samples: Number of samples in dataset
            seq_len: Sequence length for each sample
            vocab_size: Vocabulary size
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate deterministic data for reproducible tests
        torch.manual_seed(42)
        self.data = []
        for i in range(num_samples):
            input_ids = torch.randint(1, vocab_size, (seq_len,))
            attention_mask = torch.ones(seq_len)
            labels = input_ids.clone()  # Use input as labels for language modeling
            
            self.data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


class TestFullTrainingPipeline:
    """Test complete training pipeline with small datasets."""
    
    def test_basic_training_pipeline(self):
        """Test basic end-to-end training pipeline."""
        # Create small configuration for fast testing
        config = Config(
            model=MambaConfig(
                d_model=32,
                d_state=8,
                n_layers=2,
                vocab_size=100,
                pad_token_id=0
            ),
            training=TrainingConfig(
                batch_size=4,
                num_epochs=2,
                learning_rate=1e-3,
                gradient_accumulation_steps=1,
                save_steps=2,  # Save more frequently for testing
                eval_steps=2,
                max_grad_norm=1.0,
                use_mixed_precision=False
            ),
            data=DataConfig(
                max_seq_length=16,
                preprocessing_batch_size=4,
                num_workers=0  # Avoid multiprocessing in tests
            ),
            output_dir="test_output"
        )
        
        # Create model
        model = MambaModel(config.model)
        
        # Create datasets
        train_dataset = MockDataset(num_samples=20, seq_len=16, vocab_size=100)
        val_dataset = MockDataset(num_samples=8, seq_len=16, vocab_size=100)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config.data,
            distributed=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            
            # Create trainer
            trainer = DistributedTrainer(
                model=model,
                config=config,
                train_dataloader=train_loader,
                val_dataloader=val_loader
            )
            
            # Run training
            start_time = time.time()
            results = trainer.train()
            duration = time.time() - start_time
            
            # Validate results
            assert 'total_time' in results
            assert 'final_metrics' in results
            assert 'training_history' in results
            
            # Check that training progressed
            assert len(results['training_history']) == config.training.num_epochs
            
            # Check that checkpoints were saved
            checkpoint_files = list(Path(temp_dir).glob("checkpoint_epoch_*.pt"))
            assert len(checkpoint_files) > 0, "No checkpoints were saved"
            
            # Check that final checkpoint exists
            final_checkpoint = Path(temp_dir) / "final_checkpoint.pt"
            assert final_checkpoint.exists(), "Final checkpoint not found"
            
            print(f"✓ Basic training pipeline test passed ({duration:.2f}s)")
    
    def test_training_with_checkpointing(self):
        """Test training with intermediate checkpointing and resumption."""
        config = Config(
            model=MambaConfig(d_model=32, d_state=8, n_layers=2, vocab_size=50),
            training=TrainingConfig(
                batch_size=2,
                num_epochs=4,
                learning_rate=1e-3,
                save_steps=5,  # Save frequently for testing
                eval_steps=3
            ),
            data=DataConfig(max_seq_length=12, num_workers=0),
            output_dir="test_output"
        )
        
        # Create datasets
        train_dataset = MockDataset(num_samples=16, seq_len=12, vocab_size=50)
        val_dataset = MockDataset(num_samples=6, seq_len=12, vocab_size=50)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            
            # First training run (partial)
            model1 = MambaModel(config.model)
            train_loader1, val_loader1 = create_data_loaders(
                train_dataset, val_dataset, config.data, distributed=False
            )
            
            trainer1 = DistributedTrainer(
                model=model1,
                config=config,
                train_dataloader=train_loader1,
                val_dataloader=val_loader1
            )
            
            # Train for 2 epochs
            config.training.num_epochs = 2
            results1 = trainer1.train()
            
            # Check intermediate checkpoint exists
            checkpoint_files = list(Path(temp_dir).glob("checkpoint-*.pt"))
            assert len(checkpoint_files) > 0, "No intermediate checkpoints found"
            
            # Get the latest checkpoint
            latest_checkpoint = Path(temp_dir) / "latest_checkpoint.pt"
            assert latest_checkpoint.exists(), "Latest checkpoint symlink not found"
            
            # Second training run (resume from checkpoint)
            model2 = MambaModel(config.model)
            train_loader2, val_loader2 = create_data_loaders(
                train_dataset, val_dataset, config.data, distributed=False
            )
            
            # Resume training
            config.training.num_epochs = 4  # Continue to epoch 4
            trainer2 = DistributedTrainer(
                model=model2,
                config=config,
                train_dataloader=train_loader2,
                val_dataloader=val_loader2,
                resume_from_checkpoint=str(latest_checkpoint.resolve())
            )
            
            results2 = trainer2.train()
            
            # Validate resumption worked
            assert trainer2.current_epoch >= 2, "Training did not resume from correct epoch"
            assert len(results2['training_history']) == 4, "Training history not preserved"
            
            print("✓ Training with checkpointing test passed")
    
    def test_training_convergence_toy_task(self):
        """Test that training converges on a simple toy task."""
        # Create a very simple task: predict next token in a short sequence
        config = Config(
            model=MambaConfig(d_model=64, d_state=16, n_layers=3, vocab_size=20),
            training=TrainingConfig(
                batch_size=8,
                num_epochs=10,
                learning_rate=5e-4,
                gradient_accumulation_steps=1
            ),
            data=DataConfig(max_seq_length=8, num_workers=0),
            output_dir="test_output"
        )
        
        # Create simple pattern dataset (arithmetic sequences)
        def create_pattern_dataset(num_samples: int) -> MockDataset:
            """Create dataset with simple arithmetic patterns."""
            torch.manual_seed(123)  # For reproducibility
            data = []
            
            for _ in range(num_samples):
                # Create arithmetic sequence: start, start+step, start+2*step, ...
                start = torch.randint(1, 5, (1,)).item()
                step = torch.randint(1, 3, (1,)).item()
                
                sequence = []
                for i in range(8):
                    val = start + i * step
                    # Clamp to vocabulary range
                    val = min(val, 19)
                    sequence.append(val)
                
                input_ids = torch.tensor(sequence)
                attention_mask = torch.ones(8)
                # Labels are shifted by one position for next-token prediction
                labels = torch.cat([input_ids[1:], torch.tensor([0])])  # Pad last position
                
                data.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })
            
            dataset = MockDataset(num_samples=0, seq_len=8, vocab_size=20)
            dataset.data = data
            dataset.num_samples = len(data)
            return dataset
        
        train_dataset = create_pattern_dataset(100)
        val_dataset = create_pattern_dataset(20)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            
            model = MambaModel(config.model)
            train_loader, val_loader = create_data_loaders(
                train_dataset, val_dataset, config.data, distributed=False
            )
            
            trainer = DistributedTrainer(
                model=model,
                config=config,
                train_dataloader=train_loader,
                val_dataloader=val_loader
            )
            
            results = trainer.train()
            
            # Check convergence
            training_history = results['training_history']
            initial_loss = training_history[0]['train']['loss']
            final_loss = training_history[-1]['train']['loss']
            
            # Loss should decrease significantly
            loss_reduction = (initial_loss - final_loss) / initial_loss
            assert loss_reduction > 0.2, f"Insufficient convergence: {loss_reduction:.3f}"
            
            # Validation loss should also improve
            if 'val' in training_history[-1] and training_history[-1]['val']:
                initial_val_loss = training_history[0]['val']['loss']
                final_val_loss = training_history[-1]['val']['loss']
                val_loss_reduction = (initial_val_loss - final_val_loss) / initial_val_loss
                assert val_loss_reduction > 0.1, f"Validation loss did not improve: {val_loss_reduction:.3f}"
            
            print(f"✓ Training convergence test passed (loss reduction: {loss_reduction:.3f})")
    
    def test_memory_efficient_training(self):
        """Test training with memory optimization features."""
        config = Config(
            model=MambaConfig(d_model=128, d_state=32, n_layers=4, vocab_size=1000),
            training=TrainingConfig(
                batch_size=2,  # Small batch for memory efficiency
                num_epochs=2,
                learning_rate=1e-3,
                gradient_accumulation_steps=4,  # Simulate larger batch
                use_mixed_precision=True,
                max_grad_norm=1.0
            ),
            data=DataConfig(max_seq_length=64, num_workers=0),
            output_dir="test_output"
        )
        
        train_dataset = MockDataset(num_samples=32, seq_len=64, vocab_size=1000)
        val_dataset = MockDataset(num_samples=8, seq_len=64, vocab_size=1000)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            
            model = MambaModel(config.model)
            train_loader, val_loader = create_data_loaders(
                train_dataset, val_dataset, config.data, distributed=False
            )
            
            # Monitor memory usage if CUDA is available
            initial_memory = 0
            peak_memory = 0
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
            
            trainer = DistributedTrainer(
                model=model,
                config=config,
                train_dataloader=train_loader,
                val_dataloader=val_loader
            )
            
            results = trainer.train()
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_increase = (peak_memory - initial_memory) / (1024**2)  # MB
                
                # Memory usage should be reasonable for this model size
                assert memory_increase < 2000, f"Memory usage too high: {memory_increase:.1f}MB"
                
                print(f"✓ Memory efficient training test passed (peak memory: {memory_increase:.1f}MB)")
            else:
                print("✓ Memory efficient training test passed (CPU mode)")


class TestDistributedTraining:
    """Test distributed training validation across multiple processes."""
    
    def test_distributed_training_setup(self):
        """Test distributed training setup and configuration."""
        # This test validates the setup without actually running distributed training
        config = Config(
            model=MambaConfig(d_model=32, d_state=8, n_layers=2, vocab_size=50),
            training=TrainingConfig(batch_size=4, num_epochs=1, learning_rate=1e-3),
            data=DataConfig(max_seq_length=16, num_workers=0),
            output_dir="test_output"
        )
        
        model = MambaModel(config.model)
        train_dataset = MockDataset(num_samples=16, seq_len=16, vocab_size=50)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            
            train_loader, _ = create_data_loaders(
                train_dataset, None, config.data, distributed=False
            )
            
            # Test trainer initialization (single process)
            trainer = DistributedTrainer(
                model=model,
                config=config,
                train_dataloader=train_loader
            )
            
            # Validate trainer setup
            assert trainer.world_size == 1, "World size should be 1 for single process"
            assert trainer.rank == 0, "Rank should be 0 for single process"
            assert not trainer.is_distributed, "Should not be in distributed mode"
            
            # Test that model is properly wrapped
            assert isinstance(trainer.model, nn.Module)
            
            print("✓ Distributed training setup test passed")
    
    def test_distributed_sampler_functionality(self):
        """Test distributed sampler behavior."""
        from mamba_training.data.data_loader import DistributedBatchSampler, DynamicBatchSampler
        
        # Create mock dataset
        dataset = MockDataset(num_samples=20, seq_len=16, vocab_size=50)
        
        # Create base sampler
        base_sampler = DynamicBatchSampler(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            drop_last=False
        )
        
        # Test distributed sampler with simulated multi-process setup
        distributed_sampler = DistributedBatchSampler(
            sampler=base_sampler,
            num_replicas=2,
            rank=0,
            shuffle=True,
            seed=42
        )
        
        # Get batches for rank 0
        batches_rank_0 = list(distributed_sampler)
        
        # Simulate rank 1
        distributed_sampler_rank_1 = DistributedBatchSampler(
            sampler=base_sampler,
            num_replicas=2,
            rank=1,
            shuffle=True,
            seed=42
        )
        
        batches_rank_1 = list(distributed_sampler_rank_1)
        
        # Validate that batches are different between ranks
        assert len(batches_rank_0) > 0, "Rank 0 should have batches"
        assert len(batches_rank_1) > 0, "Rank 1 should have batches"
        
        # Flatten batch indices to check for overlap
        indices_rank_0 = set()
        for batch in batches_rank_0:
            indices_rank_0.update(batch)
        
        indices_rank_1 = set()
        for batch in batches_rank_1:
            indices_rank_1.update(batch)
        
        # There should be minimal overlap (due to padding for equal distribution)
        overlap = indices_rank_0.intersection(indices_rank_1)
        total_unique = len(indices_rank_0.union(indices_rank_1))
        
        # Most indices should be unique to each rank
        assert len(overlap) / total_unique < 0.5, "Too much overlap between distributed ranks"
        
        print("✓ Distributed sampler functionality test passed")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_training(self):
        """Test mixed precision training functionality."""
        config = Config(
            model=MambaConfig(d_model=64, d_state=16, n_layers=2, vocab_size=100),
            training=TrainingConfig(
                batch_size=4,
                num_epochs=1,
                learning_rate=1e-3,
                use_mixed_precision=True
            ),
            data=DataConfig(max_seq_length=32, num_workers=0),
            output_dir="test_output"
        )
        
        train_dataset = MockDataset(num_samples=16, seq_len=32, vocab_size=100)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            
            model = MambaModel(config.model).cuda()
            train_loader, _ = create_data_loaders(
                train_dataset, None, config.data, distributed=False
            )
            
            trainer = DistributedTrainer(
                model=model,
                config=config,
                train_dataloader=train_loader
            )
            
            # Run one training step to test mixed precision
            batch = next(iter(train_loader))
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            step_metrics = trainer.train_step(batch)
            
            # Validate that training step completed successfully
            assert 'loss' in step_metrics
            assert 'optimizer_step' in step_metrics
            assert not torch.isnan(torch.tensor(step_metrics['loss']))
            
            print("✓ Mixed precision training test passed")


class TestCheckpointRecovery:
    """Test checkpoint recovery and model loading functionality."""
    
    def test_checkpoint_save_load_cycle(self):
        """Test complete checkpoint save and load cycle."""
        config = Config(
            model=MambaConfig(d_model=32, d_state=8, n_layers=2, vocab_size=50),
            training=TrainingConfig(batch_size=4, num_epochs=2, learning_rate=1e-3),
            data=DataConfig(max_seq_length=16, num_workers=0),
            output_dir="test_output"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir, config)
            
            # Create model and optimizer
            model = MambaModel(config.model)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            
            # Save initial checkpoint
            initial_checkpoint_path = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                global_step=100,
                loss=2.5,
                metrics={'accuracy': 0.75},
                is_best=True
            )
            
            # Modify model state
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            
            # Create new model and optimizer
            new_model = MambaModel(config.model)
            new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
            new_scheduler = optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)
            
            # Load checkpoint
            checkpoint_data = checkpoint_manager.load_checkpoint(
                checkpoint_path=initial_checkpoint_path,
                model=new_model,
                optimizer=new_optimizer,
                scheduler=new_scheduler
            )
            
            # Validate loaded data
            assert checkpoint_data['checkpoint_data']['epoch'] == 1
            assert checkpoint_data['checkpoint_data']['global_step'] == 100
            assert abs(checkpoint_data['checkpoint_data']['loss'] - 2.5) < 1e-6
            assert checkpoint_data['checkpoint_data']['metrics']['accuracy'] == 0.75
            
            # Validate model state was loaded correctly
            original_params = torch.cat([p.flatten() for p in model.parameters()])
            loaded_params = torch.cat([p.flatten() for p in new_model.parameters()])
            
            assert torch.allclose(original_params, loaded_params, atol=1e-6), \
                "Model parameters not loaded correctly"
            
            print("✓ Checkpoint save/load cycle test passed")
    
    def test_checkpoint_validation(self):
        """Test checkpoint validation and integrity checking."""
        config = Config(
            model=MambaConfig(d_model=32, d_state=8, n_layers=2, vocab_size=50),
            training=TrainingConfig(batch_size=4, num_epochs=1, learning_rate=1e-3),
            data=DataConfig(max_seq_length=16, num_workers=0),
            output_dir="test_output"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir, config)
            
            model = MambaModel(config.model)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # Save valid checkpoint
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=0,
                global_step=50,
                loss=3.0,
                is_best=False
            )
            
            # Validate checkpoint
            validation_result = checkpoint_manager.validate_checkpoint(checkpoint_path)
            
            assert validation_result.is_valid, f"Valid checkpoint failed validation: {validation_result.validation_errors}"
            assert validation_result.epoch == 0
            assert validation_result.global_step == 50
            
            # Test validation of non-existent checkpoint
            fake_path = Path(temp_dir) / "nonexistent.pt"
            validation_result_fake = checkpoint_manager.validate_checkpoint(fake_path)
            
            assert not validation_result_fake.is_valid
            assert len(validation_result_fake.validation_errors) > 0
            
            print("✓ Checkpoint validation test passed")
    
    def test_checkpoint_cleanup(self):
        """Test checkpoint cleanup functionality."""
        config = Config(
            model=MambaConfig(d_model=16, d_state=4, n_layers=1, vocab_size=20),
            training=TrainingConfig(batch_size=2, num_epochs=1, learning_rate=1e-3),
            data=DataConfig(max_seq_length=8, num_workers=0),
            output_dir="test_output"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir, config)
            checkpoint_manager.max_checkpoints = 3  # Keep only 3 checkpoints
            
            model = MambaModel(config.model)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # Save multiple checkpoints
            checkpoint_paths = []
            for i in range(6):  # Save more than max_checkpoints
                path = checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=i,
                    global_step=i * 10,
                    loss=3.0 - i * 0.1,
                    is_best=(i == 5)  # Last one is best
                )
                checkpoint_paths.append(path)
            
            # List remaining checkpoints
            remaining_checkpoints = checkpoint_manager.list_checkpoints()
            
            # Should have kept only max_checkpoints + best
            assert len(remaining_checkpoints) <= checkpoint_manager.max_checkpoints + 1
            
            # Best checkpoint should still exist
            best_checkpoint = checkpoint_manager.get_best_checkpoint()
            assert best_checkpoint is not None
            assert best_checkpoint.exists()
            
            print("✓ Checkpoint cleanup test passed")
    
    def test_training_interruption_recovery(self):
        """Test recovery from training interruption."""
        config = Config(
            model=MambaConfig(d_model=32, d_state=8, n_layers=2, vocab_size=50),
            training=TrainingConfig(
                batch_size=4,
                num_epochs=4,
                learning_rate=1e-3,
                save_steps=5
            ),
            data=DataConfig(max_seq_length=16, num_workers=0),
            output_dir="test_output"
        )
        
        train_dataset = MockDataset(num_samples=20, seq_len=16, vocab_size=50)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            
            # First training run (simulate interruption after 2 epochs)
            model1 = MambaModel(config.model)
            train_loader1, _ = create_data_loaders(
                train_dataset, None, config.data, distributed=False
            )
            
            trainer1 = DistributedTrainer(
                model=model1,
                config=config,
                train_dataloader=train_loader1
            )
            
            # Train for 2 epochs only
            original_epochs = config.training.num_epochs
            config.training.num_epochs = 2
            results1 = trainer1.train()
            
            # Store state after interruption
            interrupted_loss = results1['final_metrics']['train']['loss']
            interrupted_step = trainer1.global_step
            
            # Second training run (resume from checkpoint)
            model2 = MambaModel(config.model)
            train_loader2, _ = create_data_loaders(
                train_dataset, None, config.data, distributed=False
            )
            
            # Find latest checkpoint
            latest_checkpoint = Path(temp_dir) / "latest_checkpoint.pt"
            assert latest_checkpoint.exists(), "Latest checkpoint not found after interruption"
            
            # Resume training
            config.training.num_epochs = original_epochs
            trainer2 = DistributedTrainer(
                model=model2,
                config=config,
                train_dataloader=train_loader2,
                resume_from_checkpoint=str(latest_checkpoint.resolve())
            )
            
            # Validate resumption state
            assert trainer2.current_epoch == 2, f"Expected epoch 2, got {trainer2.current_epoch}"
            assert trainer2.global_step == interrupted_step, f"Global step not restored correctly"
            
            # Continue training
            results2 = trainer2.train()
            
            # Validate that training continued properly
            assert len(results2['training_history']) == original_epochs
            final_loss = results2['final_metrics']['train']['loss']
            
            # Loss should continue to improve (or at least not get much worse)
            loss_change = (final_loss - interrupted_loss) / interrupted_loss
            assert loss_change < 0.5, f"Loss degraded too much after resumption: {loss_change:.3f}"
            
            print("✓ Training interruption recovery test passed")


class TestDocumentationExamples:
    """Test documentation and usage examples."""
    
    def test_basic_usage_example(self):
        """Test basic usage example from documentation."""
        # This test validates the basic usage pattern documented for users
        
        # Step 1: Create configuration
        config = Config(
            model=MambaConfig(
                d_model=64,
                d_state=16,
                n_layers=4,
                vocab_size=1000
            ),
            training=TrainingConfig(
                batch_size=8,
                num_epochs=3,
                learning_rate=1e-4
            ),
            data=DataConfig(
                max_seq_length=128,
                num_workers=0
            ),
            output_dir="example_output"
        )
        
        # Step 2: Create model
        model = MambaModel(config.model)
        
        # Step 3: Create datasets (mock for testing)
        train_dataset = MockDataset(num_samples=32, seq_len=128, vocab_size=1000)
        val_dataset = MockDataset(num_samples=8, seq_len=128, vocab_size=1000)
        
        # Step 4: Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config.data,
            distributed=False
        )
        
        # Step 5: Create trainer
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            
            trainer = DistributedTrainer(
                model=model,
                config=config,
                train_dataloader=train_loader,
                val_dataloader=val_loader
            )
            
            # Step 6: Train model
            results = trainer.train()
            
            # Validate that all expected components work together
            assert results is not None
            assert 'training_history' in results
            assert len(results['training_history']) == config.training.num_epochs
            
            print("✓ Basic usage example test passed")
    
    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test invalid model configuration
        with pytest.raises((ValueError, AssertionError)):
            invalid_config = MambaConfig(
                d_model=0,  # Invalid: must be positive
                d_state=16,
                n_layers=4,
                vocab_size=1000
            )
            model = MambaModel(invalid_config)
        
        # Test invalid training configuration
        with pytest.raises((ValueError, AssertionError)):
            invalid_training_config = TrainingConfig(
                batch_size=0,  # Invalid: must be positive
                num_epochs=1,
                learning_rate=1e-4
            )
        
        # Test valid configuration
        valid_config = Config(
            model=MambaConfig(d_model=32, d_state=8, n_layers=2, vocab_size=100),
            training=TrainingConfig(batch_size=4, num_epochs=1, learning_rate=1e-3),
            data=DataConfig(max_seq_length=16, num_workers=0),
            output_dir="test_output"
        )
        
        # Should not raise any errors
        model = MambaModel(valid_config.model)
        assert model is not None
        
        print("✓ Configuration validation test passed")
    
    def test_inference_example(self):
        """Test inference usage example."""
        # Create and train a small model
        config = Config(
            model=MambaConfig(d_model=32, d_state=8, n_layers=2, vocab_size=50),
            training=TrainingConfig(batch_size=4, num_epochs=1, learning_rate=1e-3),
            data=DataConfig(max_seq_length=16, num_workers=0),
            output_dir="test_output"
        )
        
        model = MambaModel(config.model)
        train_dataset = MockDataset(num_samples=16, seq_len=16, vocab_size=50)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            
            # Quick training
            train_loader, _ = create_data_loaders(
                train_dataset, None, config.data, distributed=False
            )
            
            trainer = DistributedTrainer(
                model=model,
                config=config,
                train_dataloader=train_loader
            )
            
            trainer.train()
            
            # Test inference
            model.eval()
            
            # Create input for generation
            prompt = torch.randint(1, config.model.vocab_size, (1, 5))
            
            with torch.no_grad():
                # Test greedy generation
                generated = model.generate(
                    input_ids=prompt,
                    max_length=15,
                    do_sample=False
                )
                
                assert generated.shape[0] == 1, "Batch size should be preserved"
                assert generated.shape[1] >= 5, "Generated sequence should be at least as long as prompt"
                assert generated.shape[1] <= 15, "Generated sequence should not exceed max_length"
                
                # Test sampling generation
                sampled = model.generate(
                    input_ids=prompt,
                    max_length=15,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9
                )
                
                assert sampled.shape[0] == 1
                assert sampled.shape[1] >= 5
                assert sampled.shape[1] <= 15
            
            print("✓ Inference example test passed")


def run_integration_tests():
    """Run all integration and end-to-end tests."""
    print("Running comprehensive integration and end-to-end tests...")
    
    # Full training pipeline tests
    print("\n=== Full Training Pipeline Tests ===")
    pipeline_test = TestFullTrainingPipeline()
    pipeline_test.test_basic_training_pipeline()
    pipeline_test.test_training_with_checkpointing()
    pipeline_test.test_training_convergence_toy_task()
    pipeline_test.test_memory_efficient_training()
    
    # Distributed training tests
    print("\n=== Distributed Training Tests ===")
    distributed_test = TestDistributedTraining()
    distributed_test.test_distributed_training_setup()
    distributed_test.test_distributed_sampler_functionality()
    if torch.cuda.is_available():
        distributed_test.test_mixed_precision_training()
    
    # Checkpoint recovery tests
    print("\n=== Checkpoint Recovery Tests ===")
    checkpoint_test = TestCheckpointRecovery()
    checkpoint_test.test_checkpoint_save_load_cycle()
    checkpoint_test.test_checkpoint_validation()
    checkpoint_test.test_checkpoint_cleanup()
    checkpoint_test.test_training_interruption_recovery()
    
    # Documentation examples tests
    print("\n=== Documentation Examples Tests ===")
    docs_test = TestDocumentationExamples()
    docs_test.test_basic_usage_example()
    docs_test.test_configuration_validation()
    docs_test.test_inference_example()
    
    print("\n✓ All integration and end-to-end tests completed successfully!")


if __name__ == "__main__":
    run_integration_tests()