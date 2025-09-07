# MambaV1
LLM Mamba Training, Inference

## Mamba Training Pipeline

A complete implementation of Mamba (State Space Model) training infrastructure from scratch, designed for efficiency, scalability, and compatibility with modern ML frameworks.

## Project Structure

```
mamba_training/
├── __init__.py                 # Main package initialization
├── config.py                   # Configuration classes and utilities
├── models/                     # Model architecture components
│   └── __init__.py
├── training/                   # Training infrastructure
│   └── __init__.py
├── data/                       # Data processing and loading
│   └── __init__.py
└── utils/                      # Utility functions
    └── __init__.py

configs/
└── default.yaml               # Default configuration file

requirements.txt               # Python dependencies
test_config.py                # Configuration system test
```

## Configuration System

The project uses a hierarchical configuration system with three main components:

### MambaConfig
Model architecture parameters:
- `d_model`: Model dimension (default: 2048)
- `d_state`: State space dimension (default: 16)
- `d_conv`: Convolution kernel size (default: 4)
- `expand`: Expansion factor (default: 2)
- `n_layers`: Number of layers (default: 24)
- `vocab_size`: Vocabulary size (default: 50280)
- `pad_token_id`: Padding token ID (default: 0)

### TrainingConfig
Training parameters:
- `batch_size`: Training batch size (default: 32)
- `gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `learning_rate`: Learning rate (default: 1e-4)
- `weight_decay`: Weight decay (default: 0.1)
- `max_grad_norm`: Maximum gradient norm (default: 1.0)
- `num_epochs`: Number of training epochs (default: 3)
- `warmup_steps`: Warmup steps (default: 1000)
- `save_steps`: Checkpoint save interval (default: 1000)
- `eval_steps`: Evaluation interval (default: 500)
- `use_mixed_precision`: Enable mixed precision (default: True)
- `gradient_checkpointing`: Enable gradient checkpointing (default: False)

### DataConfig
Data processing parameters:
- `max_seq_length`: Maximum sequence length (default: 2048)
- `tokenizer_path`: Path to tokenizer (default: "tokenizer.model")
- `dataset_path`: Path to dataset (default: "data/")
- `num_workers`: Number of data loading workers (default: 4)
- `preprocessing_batch_size`: Preprocessing batch size (default: 1000)
- `train_split`: Training data split (default: 0.9)
- `val_split`: Validation data split (default: 0.1)
- `shuffle_data`: Shuffle data (default: True)

## Usage

### Loading Configuration

```python
from mamba_training.config import ConfigLoader, create_default_config

# Create default configuration
config = create_default_config()

# Load from YAML file
config = ConfigLoader.load_from_file("configs/default.yaml")

# Load from JSON file
config = ConfigLoader.load_from_file("configs/my_config.json")
```

### Saving Configuration

```python
from mamba_training.config import ConfigLoader

# Save configuration to file
ConfigLoader.save_to_file(config, "my_experiment.yaml")
```

### Configuration Validation

```python
from mamba_training.config import validate_config_compatibility

# Validate configuration for compatibility
validate_config_compatibility(config)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Test the configuration system:
```bash
python test_config.py
```

## Requirements Addressed

This implementation addresses the following requirements from the specification:

- **Requirement 5.1**: Configurable model hyperparameters and training settings through YAML/JSON configuration files
- **Requirement 5.2**: Parameter validation and range checking with backward compatibility support

## Next Steps

The project structure is now ready for implementing the core Mamba architecture components, training infrastructure, data processing pipeline, and inference engine as outlined in the remaining tasks.
