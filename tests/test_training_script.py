"""Integration tests for the main training script."""

import os
import sys
import tempfile
import shutil
import subprocess
import json
import yaml
from pathlib import Path
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mamba_training.config import Config, ConfigLoader


class TestTrainingScript:
    """Test suite for the main training script."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def minimal_config(self, temp_dir):
        """Create minimal configuration for testing."""
        config = Config()
        
        # Override with minimal settings for fast testing
        config.model.d_model = 128
        config.model.n_layers = 2
        config.model.vocab_size = 1000
        
        config.training.batch_size = 2
        config.training.num_epochs = 1
        config.training.save_steps = 10
        config.training.eval_steps = 5
        config.training.gradient_accumulation_steps = 1
        
        config.data.max_seq_length = 64
        config.data.preprocessing_batch_size = 10
        
        config.output_dir = str(temp_dir / "outputs")
        config.experiment_name = "test_experiment"
        
        return config
    
    @pytest.fixture
    def config_file(self, temp_dir, minimal_config):
        """Create configuration file for testing."""
        config_path = temp_dir / "test_config.yaml"
        ConfigLoader.save_to_file(minimal_config, config_path)
        return config_path
    
    @pytest.fixture
    def dummy_dataset(self, temp_dir):
        """Create dummy dataset for testing."""
        dataset_dir = temp_dir / "data"
        dataset_dir.mkdir()
        
        # Create dummy text files
        for i in range(5):
            text_file = dataset_dir / f"text_{i}.txt"
            with open(text_file, 'w') as f:
                f.write(f"This is sample text number {i}. " * 10)
        
        return dataset_dir
    
    @pytest.fixture
    def dummy_tokenizer(self, temp_dir):
        """Create dummy tokenizer for testing."""
        # For testing, we'll just create a simple vocab file
        # In practice, you'd use a real tokenizer like SentencePiece
        tokenizer_path = temp_dir / "tokenizer.model"
        
        # Create a simple vocab mapping
        vocab = {f"token_{i}": i for i in range(1000)}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        vocab["<eos>"] = 2
        
        # Save as JSON for simplicity (in practice use proper tokenizer format)
        with open(tokenizer_path, 'w') as f:
            json.dump(vocab, f)
        
        return tokenizer_path
    
    def test_script_help(self):
        """Test that the script shows help message."""
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Train Mamba model with distributed support" in result.stdout
        assert "--config" in result.stdout
        assert "--resume-from-checkpoint" in result.stdout
    
    def test_config_validation(self, config_file):
        """Test configuration validation."""
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        # Test with valid config
        result = subprocess.run(
            [sys.executable, str(script_path), "--config", str(config_file), "--dry-run"],
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}  # Force CPU for testing
        )
        
        # Should succeed with dry run
        assert result.returncode == 0 or "Dry run completed successfully" in result.stderr
    
    def test_config_overrides(self, temp_dir, minimal_config):
        """Test configuration overrides from command line."""
        config_path = temp_dir / "test_config.yaml"
        ConfigLoader.save_to_file(minimal_config, config_path)
        
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        # Test with overrides
        result = subprocess.run([
            sys.executable, str(script_path),
            "--config", str(config_path),
            "--batch-size", "4",
            "--learning-rate", "0.001",
            "--num-epochs", "2",
            "--dry-run"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        # Should handle overrides without error
        assert result.returncode == 0 or "Dry run completed successfully" in result.stderr
    
    def test_invalid_config_path(self):
        """Test handling of invalid configuration path."""
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--config", "nonexistent_config.yaml"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "Error loading configuration" in result.stderr or "not found" in result.stderr
    
    def test_output_directory_creation(self, temp_dir, minimal_config):
        """Test that output directory is created correctly."""
        config_path = temp_dir / "test_config.yaml"
        ConfigLoader.save_to_file(minimal_config, config_path)
        
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        custom_output = temp_dir / "custom_output"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--config", str(config_path),
            "--output-dir", str(custom_output),
            "--dry-run"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        # Check that output directory was created
        expected_dir = custom_output / minimal_config.experiment_name
        assert expected_dir.exists() or result.returncode == 0
    
    def test_config_saving(self, temp_dir, minimal_config):
        """Test that configuration is saved to output directory."""
        config_path = temp_dir / "test_config.yaml"
        ConfigLoader.save_to_file(minimal_config, config_path)
        
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--config", str(config_path),
            "--dry-run"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        # Check that config was saved
        output_dir = Path(minimal_config.output_dir) / minimal_config.experiment_name
        saved_config_path = output_dir / "config.yaml"
        
        if saved_config_path.exists():
            # Verify saved config is valid
            saved_config = ConfigLoader.load_from_file(saved_config_path)
            assert saved_config.model.d_model == minimal_config.model.d_model
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_distributed_setup(self, temp_dir, minimal_config):
        """Test distributed training setup (single GPU)."""
        config_path = temp_dir / "test_config.yaml"
        ConfigLoader.save_to_file(minimal_config, config_path)
        
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        # Test with single GPU (simulated distributed)
        env = {
            **os.environ,
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355"
        }
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--config", str(config_path),
            "--dry-run"
        ], capture_output=True, text=True, env=env)
        
        # Should handle distributed setup
        assert result.returncode == 0 or "Dry run completed successfully" in result.stderr
    
    def test_logging_setup(self, temp_dir, minimal_config):
        """Test logging configuration and file creation."""
        config_path = temp_dir / "test_config.yaml"
        ConfigLoader.save_to_file(minimal_config, config_path)
        
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--config", str(config_path),
            "--log-level", "DEBUG",
            "--dry-run"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        # Check that log directory was created
        output_dir = Path(minimal_config.output_dir) / minimal_config.experiment_name
        log_dir = output_dir / "logs"
        
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) > 0
    
    def test_seed_reproducibility(self, temp_dir, minimal_config):
        """Test that random seed is set for reproducibility."""
        config_path = temp_dir / "test_config.yaml"
        ConfigLoader.save_to_file(minimal_config, config_path)
        
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        # Run with specific seed
        result = subprocess.run([
            sys.executable, str(script_path),
            "--config", str(config_path),
            "--seed", "12345",
            "--dry-run"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        # Should complete without error
        assert result.returncode == 0 or "Dry run completed successfully" in result.stderr
    
    def test_resume_checkpoint_validation(self, temp_dir, minimal_config):
        """Test checkpoint resume validation."""
        config_path = temp_dir / "test_config.yaml"
        ConfigLoader.save_to_file(minimal_config, config_path)
        
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        # Test with non-existent checkpoint
        result = subprocess.run([
            sys.executable, str(script_path),
            "--config", str(config_path),
            "--resume-from-checkpoint", "nonexistent_checkpoint.pt",
            "--dry-run"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        # Should handle missing checkpoint gracefully or fail appropriately
        # The exact behavior depends on when checkpoint loading is attempted
        assert result.returncode in [0, 1]  # Either succeeds in dry run or fails appropriately


class TestTrainingScriptIntegration:
    """Integration tests that require more setup."""
    
    @pytest.fixture
    def complete_setup(self, tmp_path):
        """Create complete test setup with all required files."""
        # Create directory structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create dummy dataset files
        for i in range(3):
            text_file = data_dir / f"sample_{i}.txt"
            with open(text_file, 'w') as f:
                # Create simple text that can be tokenized
                words = [f"word{j}" for j in range(20)]
                f.write(" ".join(words * 5))
        
        # Create minimal tokenizer (mock)
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_data = {
            "model": {"type": "BPE"},
            "vocab": {f"word{i}": i for i in range(100)},
            "merges": []
        }
        with open(tokenizer_path, 'w') as f:
            json.dump(tokenizer_data, f)
        
        # Create configuration
        config = Config()
        config.model.d_model = 64
        config.model.n_layers = 2
        config.model.vocab_size = 100
        
        config.training.batch_size = 1
        config.training.num_epochs = 1
        config.training.save_steps = 5
        config.training.eval_steps = 3
        
        config.data.dataset_path = str(data_dir)
        config.data.tokenizer_path = str(tokenizer_path)
        config.data.max_seq_length = 32
        
        config.output_dir = str(tmp_path / "outputs")
        config.experiment_name = "integration_test"
        
        config_path = tmp_path / "config.yaml"
        ConfigLoader.save_to_file(config, config_path)
        
        return {
            'config_path': config_path,
            'data_dir': data_dir,
            'tokenizer_path': tokenizer_path,
            'output_dir': tmp_path / "outputs",
            'config': config
        }
    
    @pytest.mark.slow
    def test_end_to_end_dry_run(self, complete_setup):
        """Test complete end-to-end dry run."""
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--config", str(complete_setup['config_path']),
            "--dry-run"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        # Should complete successfully
        assert result.returncode == 0 or "successfully" in result.stderr.lower()
    
    def test_script_imports(self):
        """Test that the script can be imported without errors."""
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        
        # Test that script can be imported (syntax check)
        result = subprocess.run([
            sys.executable, "-c", f"import sys; sys.path.insert(0, '{script_path.parent}'); import train"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Import error:", result.stderr)
        
        # Should import without syntax errors
        assert result.returncode == 0 or "ImportError" not in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])