#!/usr/bin/env python3
"""
Main training script for Mamba models with distributed training support.

This script provides a command-line interface for training Mamba models with:
- Configuration file support (YAML/JSON)
- Distributed training setup (DDP/FSDP)
- Resume training functionality from checkpoints
- Comprehensive logging and monitoring
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from torch.utils.data import random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mamba_training.config import Config, ConfigLoader, validate_config_compatibility
from mamba_training.models.mamba_model import create_mamba_model
from mamba_training.data.dataset_processor import DatasetProcessor
from mamba_training.data.data_loader import create_data_loaders
from mamba_training.training.distributed_trainer import create_distributed_trainer
from mamba_training.training.checkpoint_manager import CheckpointManager


def setup_logging(log_level: str, output_dir: Path, rank: int = 0) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        output_dir: Output directory for log files
        rank: Process rank for distributed training
    """
    # Create logs directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup file handler
    log_file = log_dir / f"train_rank_{rank}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from other libraries
    logging.getLogger("torch.distributed").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def setup_distributed() -> Dict[str, int]:
    """Setup distributed training environment.
    
    Returns:
        Dictionary containing distributed training info
    """
    # Get distributed training parameters from environment
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    distributed_info = {
        'world_size': world_size,
        'rank': rank,
        'local_rank': local_rank,
        'is_distributed': world_size > 1
    }
    
    if distributed_info['is_distributed']:
        # Initialize distributed training
        if not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend)
        
        # Set device for current process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    
    return distributed_info


def load_and_validate_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load and validate training configuration.
    
    Args:
        config_path: Path to configuration file
        overrides: Optional configuration overrides
        
    Returns:
        Validated configuration object
    """
    # Load base configuration
    config = ConfigLoader.load_from_file(config_path)
    
    # Apply overrides if provided
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logging.warning(f"Unknown config override: {key}")
    
    # Validate configuration
    validate_config_compatibility(config)
    
    return config


def create_datasets(config: Config, distributed_info: Dict[str, int]):
    """Create training and validation datasets.
    
    Args:
        config: Training configuration
        distributed_info: Distributed training information
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logging.info("Creating datasets...")
    
    # Initialize dataset processor
    processor = DatasetProcessor(
        tokenizer_path=config.data.tokenizer_path,
        max_seq_length=config.data.max_seq_length,
        config=config.data
    )
    
    # Process dataset
    dataset = processor.process_dataset(config.data.dataset_path)
    
    # Split into train/validation
    total_size = len(dataset)
    train_size = int(config.data.train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    logging.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_model(config: Config, distributed_info: Dict[str, int]):
    """Create and initialize Mamba model.
    
    Args:
        config: Training configuration
        distributed_info: Distributed training information
        
    Returns:
        Initialized Mamba model
    """
    logging.info("Creating Mamba model...")
    
    # Create model
    model = create_mamba_model(config.model)
    
    # Log model information
    param_counts = model.count_parameters()
    memory_footprint = model.get_memory_footprint()
    
    if distributed_info['rank'] == 0:
        logging.info(f"Model parameters: {param_counts}")
        logging.info(f"Model memory footprint: {memory_footprint}")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Mamba model with distributed support")
    
    # Required arguments
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file (YAML or JSON)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment name from config"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate from config"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of epochs from config"
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps from config"
    )
    
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Override checkpoint save frequency from config"
    )
    
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Override evaluation frequency from config"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without actual training"
    )
    
    args = parser.parse_args()
    
    # Setup distributed training
    distributed_info = setup_distributed()
    
    # Prepare configuration overrides
    overrides = {}
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    if args.experiment_name:
        overrides['experiment_name'] = args.experiment_name
    if args.log_level:
        overrides['log_level'] = args.log_level
    if args.seed is not None:
        overrides['seed'] = args.seed
    if args.batch_size is not None:
        overrides['training.batch_size'] = args.batch_size
    if args.learning_rate is not None:
        overrides['training.learning_rate'] = args.learning_rate
    if args.num_epochs is not None:
        overrides['training.num_epochs'] = args.num_epochs
    if args.gradient_accumulation_steps is not None:
        overrides['training.gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.save_steps is not None:
        overrides['training.save_steps'] = args.save_steps
    if args.eval_steps is not None:
        overrides['training.eval_steps'] = args.eval_steps
    
    # Load and validate configuration
    try:
        config = load_and_validate_config(args.config, overrides)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(config.log_level, output_dir, distributed_info['rank'])
    
    # Log configuration and distributed info
    if distributed_info['rank'] == 0:
        logging.info(f"Starting training with configuration: {config}")
        logging.info(f"Distributed info: {distributed_info}")
        logging.info(f"Output directory: {output_dir}")
        
        # Save configuration to output directory
        config_save_path = output_dir / "config.yaml"
        ConfigLoader.save_to_file(config, config_save_path)
        logging.info(f"Saved configuration to: {config_save_path}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    try:
        # Create datasets
        train_dataset, val_dataset = create_datasets(config, distributed_info)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config.data,
            distributed=distributed_info['is_distributed']
        )
        
        # Create model
        model = create_model(config, distributed_info)
        
        # Dry run mode - just validate setup
        if args.dry_run:
            if distributed_info['rank'] == 0:
                logging.info("Dry run completed successfully - all components initialized")
            return
        
        # Create distributed trainer
        trainer = create_distributed_trainer(
            model=model,
            config=config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        # Start training
        if distributed_info['rank'] == 0:
            logging.info("Starting training...")
        
        training_results = trainer.train()
        
        # Log final results
        if distributed_info['rank'] == 0:
            logging.info("Training completed successfully!")
            logging.info(f"Final results: {training_results}")
            
            # Save final results
            results_path = output_dir / "training_results.json"
            with open(results_path, 'w') as f:
                json.dump(training_results, f, indent=2, default=str)
            logging.info(f"Saved training results to: {results_path}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        if distributed_info['rank'] == 0:
            import traceback
            logging.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)
    
    finally:
        # Cleanup distributed training
        if distributed_info['is_distributed']:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()