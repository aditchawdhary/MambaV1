#!/usr/bin/env python3
"""
Model conversion utilities for different deployment formats.

This script provides conversion capabilities for Mamba models including:
- ONNX export for cross-platform deployment
- TorchScript compilation for production serving
- Quantization for reduced model size
- HuggingFace format conversion for ecosystem compatibility
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mamba_training.config import Config, ConfigLoader
from mamba_training.models.mamba_model import load_mamba_model


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


class ModelConverter:
    """Model conversion utilities for different deployment formats."""
    
    def __init__(self, model: nn.Module, config: Config):
        """Initialize model converter.
        
        Args:
            model: Trained Mamba model
            config: Model configuration
        """
        self.model = model
        self.config = config
        self.model.eval()
    
    def export_to_onnx(self, 
                      output_path: str,
                      input_shape: Tuple[int, int] = (1, 512),
                      opset_version: int = 11,
                      dynamic_axes: bool = True) -> Dict[str, Any]:
        """Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Input shape (batch_size, sequence_length)
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic axes for variable input sizes
            
        Returns:
            Dictionary with export information
        """
        logging.info(f"Exporting model to ONNX: {output_path}")
        
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX export requires 'onnx' and 'onnxruntime' packages")
        
        # Create dummy input
        dummy_input = torch.randint(
            0, self.config.model.vocab_size,
            input_shape,
            dtype=torch.long
        )
        
        # Define dynamic axes if requested
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        
        # Export to ONNX
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes_dict,
            verbose=False
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        # Test ONNX runtime
        ort_session = ort.InferenceSession(str(output_path))
        
        # Run test inference
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Compare with PyTorch output
        with torch.no_grad():
            torch_output = self.model(dummy_input)
            if hasattr(torch_output, 'logits'):
                torch_output = torch_output.logits
        
        max_diff = abs(torch_output.numpy() - ort_outputs[0]).max()
        
        export_info = {
            'output_path': str(output_path),
            'input_shape': input_shape,
            'opset_version': opset_version,
            'dynamic_axes': dynamic_axes,
            'max_difference': float(max_diff),
            'model_size_mb': output_path.stat().st_size / (1024**2)
        }
        
        logging.info(f"ONNX export completed. Max difference: {max_diff:.2e}")
        logging.info(f"Model size: {export_info['model_size_mb']:.2f} MB")
        
        return export_info
    
    def export_to_torchscript(self, 
                             output_path: str,
                             method: str = "trace",
                             input_shape: Tuple[int, int] = (1, 512)) -> Dict[str, Any]:
        """Export model to TorchScript format.
        
        Args:
            output_path: Path to save TorchScript model
            method: Export method ("trace" or "script")
            input_shape: Input shape for tracing
            
        Returns:
            Dictionary with export information
        """
        logging.info(f"Exporting model to TorchScript ({method}): {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if method == "trace":
            # Create dummy input for tracing
            dummy_input = torch.randint(
                0, self.config.model.vocab_size,
                input_shape,
                dtype=torch.long
            )
            
            # Trace the model
            traced_model = torch.jit.trace(self.model, dummy_input)
            
            # Test traced model
            with torch.no_grad():
                original_output = self.model(dummy_input)
                traced_output = traced_model(dummy_input)
                
                if hasattr(original_output, 'logits'):
                    original_output = original_output.logits
                if hasattr(traced_output, 'logits'):
                    traced_output = traced_output.logits
                
                max_diff = (original_output - traced_output).abs().max().item()
            
            # Save traced model
            traced_model.save(str(output_path))
            
        elif method == "script":
            # Script the model
            scripted_model = torch.jit.script(self.model)
            
            # Test scripted model
            dummy_input = torch.randint(
                0, self.config.model.vocab_size,
                input_shape,
                dtype=torch.long
            )
            
            with torch.no_grad():
                original_output = self.model(dummy_input)
                scripted_output = scripted_model(dummy_input)
                
                if hasattr(original_output, 'logits'):
                    original_output = original_output.logits
                if hasattr(scripted_output, 'logits'):
                    scripted_output = scripted_output.logits
                
                max_diff = (original_output - scripted_output).abs().max().item()
            
            # Save scripted model
            scripted_model.save(str(output_path))
            
        else:
            raise ValueError(f"Unknown TorchScript method: {method}")
        
        export_info = {
            'output_path': str(output_path),
            'method': method,
            'input_shape': input_shape,
            'max_difference': max_diff,
            'model_size_mb': output_path.stat().st_size / (1024**2)
        }
        
        logging.info(f"TorchScript export completed. Max difference: {max_diff:.2e}")
        logging.info(f"Model size: {export_info['model_size_mb']:.2f} MB")
        
        return export_info
    
    def quantize_model(self, 
                      output_path: str,
                      quantization_type: str = "dynamic",
                      dtype: str = "qint8") -> Dict[str, Any]:
        """Quantize model for reduced size and faster inference.
        
        Args:
            output_path: Path to save quantized model
            quantization_type: Type of quantization ("dynamic" or "static")
            dtype: Quantization data type ("qint8" or "float16")
            
        Returns:
            Dictionary with quantization information
        """
        logging.info(f"Quantizing model ({quantization_type}, {dtype}): {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if dtype == "float16":
            # Half precision quantization
            quantized_model = self.model.half()
            
            # Test quantized model
            dummy_input = torch.randint(
                0, self.config.model.vocab_size,
                (1, 512),
                dtype=torch.long
            )
            
            with torch.no_grad():
                original_output = self.model.float()(dummy_input)
                quantized_output = quantized_model(dummy_input)
                
                if hasattr(original_output, 'logits'):
                    original_output = original_output.logits
                if hasattr(quantized_output, 'logits'):
                    quantized_output = quantized_output.logits.float()
                
                max_diff = (original_output - quantized_output).abs().max().item()
            
            # Save quantized model
            torch.save({
                'model_state_dict': quantized_model.state_dict(),
                'config': self.config,
                'quantization_info': {
                    'type': quantization_type,
                    'dtype': dtype
                }
            }, output_path)
            
        elif dtype == "qint8":
            # Dynamic quantization
            if quantization_type == "dynamic":
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear},
                    dtype=torch.qint8
                )
            else:
                raise NotImplementedError("Static quantization not implemented yet")
            
            # Test quantized model
            dummy_input = torch.randint(
                0, self.config.model.vocab_size,
                (1, 512),
                dtype=torch.long
            )
            
            with torch.no_grad():
                original_output = self.model(dummy_input)
                quantized_output = quantized_model(dummy_input)
                
                if hasattr(original_output, 'logits'):
                    original_output = original_output.logits
                if hasattr(quantized_output, 'logits'):
                    quantized_output = quantized_output.logits
                
                max_diff = (original_output - quantized_output).abs().max().item()
            
            # Save quantized model
            torch.save({
                'model': quantized_model,
                'config': self.config,
                'quantization_info': {
                    'type': quantization_type,
                    'dtype': dtype
                }
            }, output_path)
        
        else:
            raise ValueError(f"Unknown quantization dtype: {dtype}")
        
        # Calculate size reduction
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        quantized_size = output_path.stat().st_size
        size_reduction = (1 - quantized_size / original_size) * 100
        
        quantization_info = {
            'output_path': str(output_path),
            'quantization_type': quantization_type,
            'dtype': dtype,
            'max_difference': max_diff,
            'original_size_mb': original_size / (1024**2),
            'quantized_size_mb': quantized_size / (1024**2),
            'size_reduction_percent': size_reduction
        }
        
        logging.info(f"Quantization completed. Max difference: {max_diff:.2e}")
        logging.info(f"Size reduction: {size_reduction:.1f}%")
        
        return quantization_info
    
    def export_to_huggingface(self, output_dir: str) -> Dict[str, Any]:
        """Export model to HuggingFace format for ecosystem compatibility.
        
        Args:
            output_dir: Directory to save HuggingFace format files
            
        Returns:
            Dictionary with export information
        """
        logging.info(f"Exporting model to HuggingFace format: {output_dir}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        model_path = output_dir / "pytorch_model.bin"
        torch.save(self.model.state_dict(), model_path)
        
        # Create config.json
        config_dict = {
            "architectures": ["MambaModel"],
            "model_type": "mamba",
            "d_model": self.config.model.d_model,
            "d_state": self.config.model.d_state,
            "d_conv": self.config.model.d_conv,
            "expand": self.config.model.expand,
            "n_layers": self.config.model.n_layers,
            "vocab_size": self.config.model.vocab_size,
            "pad_token_id": self.config.model.pad_token_id,
            "torch_dtype": "float32",
            "transformers_version": "4.0.0"
        }
        
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Create generation_config.json
        generation_config = {
            "max_length": 512,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": self.config.model.pad_token_id,
            "eos_token_id": 2  # Assuming EOS token ID
        }
        
        gen_config_path = output_dir / "generation_config.json"
        with open(gen_config_path, 'w') as f:
            json.dump(generation_config, f, indent=2)
        
        # Create README.md
        readme_content = f"""# Mamba Model

This is a Mamba (State Space Model) trained for text generation.

## Model Details

- **Architecture**: Mamba
- **Parameters**: {self.model.count_parameters()['total']:,}
- **Hidden Size**: {self.config.model.d_model}
- **Number of Layers**: {self.config.model.n_layers}
- **Vocabulary Size**: {self.config.model.vocab_size}

## Usage

```python
import torch
from mamba_training.models.mamba_model import MambaModel
from mamba_training.config import MambaConfig

# Load configuration
config = MambaConfig(
    d_model={self.config.model.d_model},
    d_state={self.config.model.d_state},
    d_conv={self.config.model.d_conv},
    expand={self.config.model.expand},
    n_layers={self.config.model.n_layers},
    vocab_size={self.config.model.vocab_size}
)

# Load model
model = MambaModel(config)
model.load_state_dict(torch.load("pytorch_model.bin"))
model.eval()

# Generate text
input_ids = torch.tensor([[1, 2, 3]])  # Your tokenized input
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits
```

## Training Details

This model was trained using the Mamba training pipeline with the following configuration:
- Model dimension: {self.config.model.d_model}
- State dimension: {self.config.model.d_state}
- Convolution dimension: {self.config.model.d_conv}
- Expansion factor: {self.config.model.expand}
"""
        
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        export_info = {
            'output_dir': str(output_dir),
            'files_created': [
                str(model_path.name),
                str(config_path.name),
                str(gen_config_path.name),
                str(readme_path.name)
            ],
            'total_size_mb': sum(
                f.stat().st_size for f in output_dir.iterdir() if f.is_file()
            ) / (1024**2)
        }
        
        logging.info(f"HuggingFace export completed. Total size: {export_info['total_size_mb']:.2f} MB")
        
        return export_info


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert Mamba model to different formats")
    
    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for converted model"
    )
    
    # Conversion format
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["onnx", "torchscript", "quantized", "huggingface"],
        help="Target conversion format"
    )
    
    # Format-specific options
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=2,
        default=[1, 512],
        help="Input shape for ONNX/TorchScript export (batch_size seq_len)"
    )
    
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version"
    )
    
    parser.add_argument(
        "--torchscript-method",
        type=str,
        choices=["trace", "script"],
        default="trace",
        help="TorchScript export method"
    )
    
    parser.add_argument(
        "--quantization-type",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization type"
    )
    
    parser.add_argument(
        "--quantization-dtype",
        type=str,
        choices=["qint8", "float16"],
        default="qint8",
        help="Quantization data type"
    )
    
    parser.add_argument(
        "--no-dynamic-axes",
        action="store_true",
        help="Disable dynamic axes for ONNX export"
    )
    
    # Other options
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration and model
    try:
        config = ConfigLoader.load_from_file(args.config)
        model = load_mamba_model(args.model_path, config.model)
        
        logging.info(f"Loaded model from: {args.model_path}")
        param_counts = model.count_parameters()
        logging.info(f"Model parameters: {param_counts}")
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Initialize converter
    converter = ModelConverter(model, config)
    
    # Perform conversion based on format
    try:
        if args.format == "onnx":
            result = converter.export_to_onnx(
                output_path=args.output_path,
                input_shape=tuple(args.input_shape),
                opset_version=args.opset_version,
                dynamic_axes=not args.no_dynamic_axes
            )
            
        elif args.format == "torchscript":
            result = converter.export_to_torchscript(
                output_path=args.output_path,
                method=args.torchscript_method,
                input_shape=tuple(args.input_shape)
            )
            
        elif args.format == "quantized":
            result = converter.quantize_model(
                output_path=args.output_path,
                quantization_type=args.quantization_type,
                dtype=args.quantization_dtype
            )
            
        elif args.format == "huggingface":
            result = converter.export_to_huggingface(
                output_dir=args.output_path
            )
        
        # Print results
        logging.info("Conversion completed successfully!")
        logging.info(f"Results: {json.dumps(result, indent=2)}")
        
        # Save conversion info
        info_path = Path(args.output_path).parent / f"conversion_info_{args.format}.json"
        with open(info_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logging.info(f"Conversion info saved to: {info_path}")
        
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()