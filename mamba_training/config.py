"""Configuration classes for Mamba training pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import yaml
import json
from pathlib import Path


@dataclass
class MambaConfig:
    """Configuration for Mamba model architecture."""
    
    d_model: int = 2048
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 24
    vocab_size: int = 50280
    pad_token_id: int = 0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_state <= 0:
            raise ValueError("d_state must be positive")
        if self.d_conv <= 0:
            raise ValueError("d_conv must be positive")
        if self.expand <= 0:
            raise ValueError("expand must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")


@dataclass
class TrainingConfig:
    """Configuration for training parameters and optimization."""
    
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    warmup_steps: int = 1000
    save_steps: int = 1000
    eval_steps: int = 500
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        """Validate training configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.save_steps <= 0:
            raise ValueError("save_steps must be positive")
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive")


@dataclass
class DataConfig:
    """Configuration for data processing and loading."""
    
    max_seq_length: int = 2048
    tokenizer_path: str = "tokenizer.model"
    dataset_path: str = "data/"
    num_workers: int = 4
    preprocessing_batch_size: int = 1000
    train_split: float = 0.9
    val_split: float = 0.1
    shuffle_data: bool = True
    
    def __post_init__(self):
        """Validate data configuration parameters."""
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.preprocessing_batch_size <= 0:
            raise ValueError("preprocessing_batch_size must be positive")
        if not (0 < self.train_split < 1):
            raise ValueError("train_split must be between 0 and 1")
        if not (0 < self.val_split < 1):
            raise ValueError("val_split must be between 0 and 1")
        if abs(self.train_split + self.val_split - 1.0) > 1e-6:
            raise ValueError("train_split + val_split must equal 1.0")


@dataclass
class Config:
    """Main configuration class combining all component configs."""
    
    model: MambaConfig = field(default_factory=MambaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Additional metadata
    experiment_name: str = "mamba_experiment"
    output_dir: str = "outputs"
    log_level: str = "INFO"
    seed: int = 42
    
    def __post_init__(self):
        """Validate main configuration."""
        if not self.experiment_name:
            raise ValueError("experiment_name cannot be empty")
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError("log_level must be one of: DEBUG, INFO, WARNING, ERROR")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


class ConfigLoader:
    """Utility class for loading and validating configurations from files."""
    
    @staticmethod
    def load_from_file(config_path: Union[str, Path]) -> Config:
        """Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config: Loaded and validated configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid or validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration data based on file extension
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return ConfigLoader._dict_to_config(config_data)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object with validation.
        
        Args:
            config_dict: Dictionary containing configuration data
            
        Returns:
            Config: Validated configuration object
        """
        # Extract component configurations
        model_config = MambaConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        # Extract main config parameters
        main_config_params = {
            k: v for k, v in config_dict.items() 
            if k not in ['model', 'training', 'data']
        }
        
        return Config(
            model=model_config,
            training=training_config,
            data=data_config,
            **main_config_params
        )
    
    @staticmethod
    def save_to_file(config: Config, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML or JSON file.
        
        Args:
            config: Configuration object to save
            config_path: Path where to save the configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary
        config_dict = ConfigLoader._config_to_dict(config)
        
        # Save based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    @staticmethod
    def _config_to_dict(config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary.
        
        Args:
            config: Configuration object
            
        Returns:
            Dict: Configuration as dictionary
        """
        return {
            'model': {
                'd_model': config.model.d_model,
                'd_state': config.model.d_state,
                'd_conv': config.model.d_conv,
                'expand': config.model.expand,
                'n_layers': config.model.n_layers,
                'vocab_size': config.model.vocab_size,
                'pad_token_id': config.model.pad_token_id,
            },
            'training': {
                'batch_size': config.training.batch_size,
                'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
                'learning_rate': config.training.learning_rate,
                'weight_decay': config.training.weight_decay,
                'max_grad_norm': config.training.max_grad_norm,
                'num_epochs': config.training.num_epochs,
                'warmup_steps': config.training.warmup_steps,
                'save_steps': config.training.save_steps,
                'eval_steps': config.training.eval_steps,
                'use_mixed_precision': config.training.use_mixed_precision,
                'gradient_checkpointing': config.training.gradient_checkpointing,
            },
            'data': {
                'max_seq_length': config.data.max_seq_length,
                'tokenizer_path': config.data.tokenizer_path,
                'dataset_path': config.data.dataset_path,
                'num_workers': config.data.num_workers,
                'preprocessing_batch_size': config.data.preprocessing_batch_size,
                'train_split': config.data.train_split,
                'val_split': config.data.val_split,
                'shuffle_data': config.data.shuffle_data,
            },
            'experiment_name': config.experiment_name,
            'output_dir': config.output_dir,
            'log_level': config.log_level,
            'seed': config.seed,
        }


def create_default_config() -> Config:
    """Create a default configuration with sensible defaults.
    
    Returns:
        Config: Default configuration object
    """
    return Config()


def validate_config_compatibility(config: Config) -> None:
    """Validate configuration for internal consistency and compatibility.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration has incompatible settings
    """
    # Check if effective batch size is reasonable
    effective_batch_size = config.training.batch_size * config.training.gradient_accumulation_steps
    if effective_batch_size > 1024:
        raise ValueError(f"Effective batch size ({effective_batch_size}) may be too large")
    
    # Check if model dimensions are compatible
    if config.model.d_model % config.model.expand != 0:
        raise ValueError("d_model should be divisible by expand factor for optimal performance")
    
    # Check if sequence length is reasonable for model size
    if config.data.max_seq_length > 8192 and config.model.d_model > 2048:
        raise ValueError("Large model with long sequences may cause memory issues")
    
    # Validate paths exist if they should
    tokenizer_path = Path(config.data.tokenizer_path)
    if not tokenizer_path.is_absolute() and not tokenizer_path.exists():
        # Only warn for relative paths that don't exist yet
        pass
    
    dataset_path = Path(config.data.dataset_path)
    if not dataset_path.is_absolute() and not dataset_path.exists():
        # Only warn for relative paths that don't exist yet
        pass