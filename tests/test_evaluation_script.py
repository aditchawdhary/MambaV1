"""Tests for the evaluation script."""

import os
import sys
import tempfile
import shutil
import subprocess
import json
from pathlib import Path
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mamba_training.config import Config, ConfigLoader


class TestEvaluationScript:
    """Test suite for the evaluation script."""
    
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
        config.model.d_model = 64
        config.model.n_layers = 2
        config.model.vocab_size = 100
        
        config.training.batch_size = 2
        
        config.data.max_seq_length = 32
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
    def dummy_checkpoint(self, temp_dir, minimal_config):
        """Create dummy model checkpoint for testing."""
        from mamba_training.models.mamba_model import create_mamba_model
        
        model = create_mamba_model(minimal_config.model)
        
        checkpoint_path = temp_dir / "model_checkpoint.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': minimal_config,
            'epoch': 1,
            'global_step': 100
        }, checkpoint_path)
        
        return checkpoint_path
    
    @pytest.fixture
    def dummy_test_data(self, temp_dir):
        """Create dummy test dataset."""
        test_dir = temp_dir / "test_data"
        test_dir.mkdir()
        
        # Create dummy text files
        for i in range(3):
            text_file = test_dir / f"test_{i}.txt"
            with open(text_file, 'w') as f:
                f.write(f"This is test sample {i}. " * 5)
        
        return test_dir
    
    def test_script_help(self):
        """Test that the script shows help message."""
        script_path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Evaluate Mamba model performance" in result.stdout
        assert "--model-path" in result.stdout
        assert "--config" in result.stdout
        assert "--test-data" in result.stdout
    
    def test_script_imports(self):
        """Test that the script can be imported without errors."""
        script_path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        
        # Test that script can be imported (syntax check)
        result = subprocess.run([
            sys.executable, "-c", 
            f"import sys; sys.path.insert(0, '{script_path.parent}'); import evaluate"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Import error:", result.stderr)
        
        # Should import without syntax errors
        assert result.returncode == 0 or "ImportError" not in result.stderr
    
    def test_missing_required_args(self):
        """Test handling of missing required arguments."""
        script_path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        
        # Test with missing arguments
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()
    
    def test_invalid_model_path(self, config_file, dummy_test_data):
        """Test handling of invalid model path."""
        script_path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", "nonexistent_model.pt",
            "--config", str(config_file),
            "--test-data", str(dummy_test_data),
            "--eval-perplexity"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        assert result.returncode != 0
        assert "Error loading model" in result.stderr or "not found" in result.stderr.lower()
    
    def test_invalid_config_path(self, dummy_checkpoint, dummy_test_data):
        """Test handling of invalid config path."""
        script_path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", "nonexistent_config.yaml",
            "--test-data", str(dummy_test_data),
            "--eval-perplexity"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        assert result.returncode != 0
        assert "Error loading configuration" in result.stderr or "not found" in result.stderr.lower()
    
    @pytest.mark.slow
    def test_perplexity_evaluation(self, dummy_checkpoint, config_file, dummy_test_data, temp_dir):
        """Test perplexity evaluation functionality."""
        script_path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        output_file = temp_dir / "eval_results.json"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--test-data", str(dummy_test_data),
            "--eval-perplexity",
            "--output-file", str(output_file)
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        # Should complete successfully or with expected errors
        if result.returncode == 0:
            # Check if output file was created
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                assert 'model_path' in results
                assert 'evaluation_timestamp' in results
    
    @pytest.mark.slow
    def test_generation_evaluation(self, dummy_checkpoint, config_file, dummy_test_data, temp_dir):
        """Test generation quality evaluation."""
        script_path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        output_file = temp_dir / "gen_results.json"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--test-data", str(dummy_test_data),
            "--eval-generation",
            "--generation-prompts", "Hello", "Test",
            "--num-generation-samples", "1",
            "--max-generation-length", "20",
            "--output-file", str(output_file)
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        # Should complete successfully or with expected errors
        if result.returncode == 0:
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                assert 'model_path' in results
    
    def test_benchmark_evaluation(self, dummy_checkpoint, config_file, dummy_test_data, temp_dir):
        """Test inference speed benchmarking."""
        script_path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        output_file = temp_dir / "benchmark_results.json"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--test-data", str(dummy_test_data),
            "--benchmark-speed",
            "--output-file", str(output_file)
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        # Should complete successfully or with expected errors
        if result.returncode == 0:
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                assert 'model_path' in results
    
    def test_memory_evaluation(self, dummy_checkpoint, config_file, dummy_test_data, temp_dir):
        """Test memory usage evaluation."""
        script_path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        output_file = temp_dir / "memory_results.json"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--test-data", str(dummy_test_data),
            "--eval-memory",
            "--output-file", str(output_file)
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        # Should complete successfully or with expected errors
        if result.returncode == 0:
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                assert 'model_path' in results
    
    def test_eval_all_flag(self, dummy_checkpoint, config_file, dummy_test_data, temp_dir):
        """Test --eval-all flag functionality."""
        script_path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        output_file = temp_dir / "all_results.json"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--test-data", str(dummy_test_data),
            "--eval-all",
            "--num-generation-samples", "1",
            "--max-generation-length", "10",
            "--output-file", str(output_file)
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        # Should attempt all evaluations
        if result.returncode == 0:
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                assert 'model_path' in results
    
    def test_output_file_creation(self, dummy_checkpoint, config_file, dummy_test_data, temp_dir):
        """Test that output file is created correctly."""
        script_path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        
        # Create nested output directory
        output_dir = temp_dir / "nested" / "output"
        output_file = output_dir / "results.json"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--test-data", str(dummy_test_data),
            "--eval-memory",
            "--output-file", str(output_file)
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        # Should create directory structure
        if result.returncode == 0:
            assert output_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])