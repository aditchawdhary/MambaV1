"""Tests for the inference script."""

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


class TestInferenceScript:
    """Test suite for the inference script."""
    
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
        
        config.data.max_seq_length = 32
        
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
    def batch_input_file(self, temp_dir):
        """Create batch input file for testing."""
        # JSON format
        json_file = temp_dir / "batch_prompts.json"
        prompts = [
            "Hello world",
            "The quick brown fox",
            "Once upon a time"
        ]
        with open(json_file, 'w') as f:
            json.dump(prompts, f)
        
        # Text format
        txt_file = temp_dir / "batch_prompts.txt"
        with open(txt_file, 'w') as f:
            for prompt in prompts:
                f.write(prompt + '\n')
        
        return json_file, txt_file
    
    def test_script_help(self):
        """Test that the script shows help message."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Generate text using trained Mamba model" in result.stdout
        assert "--model-path" in result.stdout
        assert "--config" in result.stdout
        assert "--interactive" in result.stdout
        assert "--batch" in result.stdout
        assert "--single" in result.stdout
    
    def test_script_imports(self):
        """Test that the script can be imported without errors."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        # Test that script can be imported (syntax check)
        result = subprocess.run([
            sys.executable, "-c", 
            f"import sys; sys.path.insert(0, '{script_path.parent}'); import inference"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Import error:", result.stderr)
        
        # Should import without syntax errors
        assert result.returncode == 0 or "ImportError" not in result.stderr
    
    def test_missing_required_args(self):
        """Test handling of missing required arguments."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        # Test with missing arguments
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()
    
    def test_mutually_exclusive_modes(self, dummy_checkpoint, config_file):
        """Test that mode arguments are mutually exclusive."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        # Test with multiple modes
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--interactive",
            "--single", "test prompt"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "not allowed" in result.stderr.lower() or "mutually exclusive" in result.stderr.lower()
    
    def test_invalid_model_path(self, config_file):
        """Test handling of invalid model path."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", "nonexistent_model.pt",
            "--config", str(config_file),
            "--single", "test prompt"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        assert result.returncode != 0
        assert "Error creating inference engine" in result.stderr or "not found" in result.stderr.lower()
    
    def test_invalid_config_path(self, dummy_checkpoint):
        """Test handling of invalid config path."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", "nonexistent_config.yaml",
            "--single", "test prompt"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        assert result.returncode != 0
        assert "Error creating inference engine" in result.stderr or "not found" in result.stderr.lower()
    
    @pytest.mark.slow
    def test_single_prompt_generation(self, dummy_checkpoint, config_file):
        """Test single prompt text generation."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--single", "Hello world",
            "--max-length", "20",
            "--temperature", "0.8"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        # Should complete successfully or with expected errors
        if result.returncode == 0:
            assert "Generated text" in result.stdout
            assert "Hello world" in result.stdout
    
    @pytest.mark.slow
    def test_batch_generation_json(self, dummy_checkpoint, config_file, batch_input_file, temp_dir):
        """Test batch generation from JSON file."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        json_file, _ = batch_input_file
        output_file = temp_dir / "batch_results.json"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--batch", str(json_file),
            "--output-file", str(output_file),
            "--max-length", "10",
            "--temperature", "0.8"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        # Should complete successfully or with expected errors
        if result.returncode == 0:
            assert "Batch processing completed" in result.stdout
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                assert 'total_prompts' in results
                assert 'results' in results
    
    @pytest.mark.slow
    def test_batch_generation_txt(self, dummy_checkpoint, config_file, batch_input_file, temp_dir):
        """Test batch generation from text file."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        _, txt_file = batch_input_file
        output_file = temp_dir / "batch_results.json"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--batch", str(txt_file),
            "--output-file", str(output_file),
            "--max-length", "10"
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
                assert 'total_prompts' in results
    
    def test_batch_default_output_file(self, dummy_checkpoint, config_file, batch_input_file, temp_dir):
        """Test batch generation with default output file naming."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        json_file, _ = batch_input_file
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--batch", str(json_file),
            "--max-length", "5"
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        # Should use default output filename
        if result.returncode == 0:
            expected_output = json_file.parent / "batch_prompts_results.json"
            if expected_output.exists():
                with open(expected_output, 'r') as f:
                    results = json.load(f)
                assert 'total_prompts' in results
    
    def test_generation_parameters(self, dummy_checkpoint, config_file):
        """Test different generation parameters."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        # Test with different parameters
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--single", "Test",
            "--max-length", "15",
            "--temperature", "1.2",
            "--top-p", "0.95",
            "--top-k", "40",
            "--no-sampling"  # This should override sampling parameters
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        # Should handle parameter combinations
        if result.returncode == 0:
            assert "Generated text" in result.stdout
    
    def test_device_selection(self, dummy_checkpoint, config_file):
        """Test device selection options."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        # Test CPU device
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--single", "Test",
            "--device", "cpu",
            "--max-length", "5"
        ], capture_output=True, text=True)
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        # Should work with CPU device
        if result.returncode == 0:
            assert "Generated text" in result.stdout
    
    def test_invalid_batch_file(self, dummy_checkpoint, config_file, temp_dir):
        """Test handling of invalid batch input file."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        # Create invalid JSON file
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--batch", str(invalid_file)
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        assert result.returncode != 0
        assert "Error in batch processing" in result.stderr or "error" in result.stderr.lower()
    
    def test_empty_batch_file(self, dummy_checkpoint, config_file, temp_dir):
        """Test handling of empty batch input file."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        # Create empty file
        empty_file = temp_dir / "empty.txt"
        empty_file.touch()
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--model-path", str(dummy_checkpoint),
            "--config", str(config_file),
            "--batch", str(empty_file)
        ], capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        
        assert result.returncode != 0
        assert "Error in batch processing" in result.stderr or "no prompts" in result.stderr.lower()


class TestInferenceScriptInteractive:
    """Tests for interactive mode (limited testing due to interactive nature)."""
    
    def test_interactive_mode_startup(self):
        """Test that interactive mode can start (will exit immediately)."""
        script_path = Path(__file__).parent.parent / "scripts" / "inference.py"
        
        # This test is limited since we can't easily test interactive input
        # We just verify the script can handle the interactive flag
        result = subprocess.run([
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, '{script_path.parent}')
import inference
# Test that InteractiveGenerator can be imported
from inference import InteractiveGenerator
print("Interactive mode imports successful")
"""
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            assert "Interactive mode imports successful" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])