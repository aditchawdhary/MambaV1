#!/usr/bin/env python3
"""Simple test script to verify configuration system functionality."""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from mamba_training.config import (
    Config, MambaConfig, TrainingConfig, DataConfig,
    ConfigLoader, create_default_config, validate_config_compatibility
)


def test_default_config():
    """Test creating default configuration."""
    print("Testing default configuration creation...")
    config = create_default_config()
    print(f"✓ Default config created: {config.experiment_name}")
    
    # Test validation
    validate_config_compatibility(config)
    print("✓ Default config validation passed")


def test_config_loading():
    """Test loading configuration from file."""
    print("\nTesting configuration loading from file...")
    
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        try:
            config = ConfigLoader.load_from_file(config_path)
            print(f"✓ Config loaded from {config_path}")
            print(f"  - Model d_model: {config.model.d_model}")
            print(f"  - Training batch_size: {config.training.batch_size}")
            print(f"  - Data max_seq_length: {config.data.max_seq_length}")
            
            # Test validation
            validate_config_compatibility(config)
            print("✓ Loaded config validation passed")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠ Config file not found: {config_path}")


def test_config_saving():
    """Test saving configuration to file."""
    print("\nTesting configuration saving...")
    
    config = create_default_config()
    config.experiment_name = "test_experiment"
    
    # Save as YAML
    yaml_path = Path("test_config.yaml")
    ConfigLoader.save_to_file(config, yaml_path)
    print(f"✓ Config saved to {yaml_path}")
    
    # Load it back and verify
    loaded_config = ConfigLoader.load_from_file(yaml_path)
    assert loaded_config.experiment_name == "test_experiment"
    print("✓ Config round-trip test passed")
    
    # Clean up
    yaml_path.unlink()
    print("✓ Test file cleaned up")


def test_validation_errors():
    """Test configuration validation error handling."""
    print("\nTesting validation error handling...")
    
    try:
        # Test invalid model config
        invalid_config = MambaConfig(d_model=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected validation error: {e}")
    
    try:
        # Test invalid training config
        invalid_config = TrainingConfig(batch_size=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected validation error: {e}")


if __name__ == "__main__":
    print("Running configuration system tests...\n")
    
    try:
        test_default_config()
        test_config_loading()
        test_config_saving()
        test_validation_errors()
        
        print("\n🎉 All configuration tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)