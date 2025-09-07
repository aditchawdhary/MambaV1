#!/usr/bin/env python3
"""
Inference script for text generation using trained Mamba models.

This script provides interactive and batch text generation capabilities including:
- Interactive text generation with customizable parameters
- Batch inference from file inputs
- Multiple sampling strategies (greedy, temperature, top-p, top-k)
- Performance monitoring and optimization
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mamba_training.config import Config, ConfigLoader
from mamba_training.models.mamba_model import load_mamba_model
from mamba_training.inference.mamba_inference import MambaInference


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


class InteractiveGenerator:
    """Interactive text generation interface."""
    
    def __init__(self, inference_engine: MambaInference):
        """Initialize interactive generator.
        
        Args:
            inference_engine: Configured MambaInference engine
        """
        self.inference_engine = inference_engine
        self.generation_history = []
    
    def run_interactive_session(self) -> None:
        """Run interactive text generation session."""
        print("\n" + "="*60)
        print("🤖 Mamba Interactive Text Generation")
        print("="*60)
        print("Commands:")
        print("  /help     - Show this help message")
        print("  /settings - Show current generation settings")
        print("  /set <param> <value> - Change generation parameter")
        print("  /history  - Show generation history")
        print("  /clear    - Clear generation history")
        print("  /quit     - Exit interactive mode")
        print("  <text>    - Generate text from prompt")
        print("="*60)
        
        # Default generation parameters
        gen_params = {
            'max_length': 100,
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 50,
            'do_sample': True
        }
        
        while True:
            try:
                # Get user input
                prompt = input("\n🔤 Enter prompt (or command): ").strip()
                
                if not prompt:
                    continue
                
                # Handle commands
                if prompt.startswith('/'):
                    if prompt == '/help':
                        self._show_help()
                    elif prompt == '/settings':
                        self._show_settings(gen_params)
                    elif prompt.startswith('/set '):
                        self._handle_set_command(prompt, gen_params)
                    elif prompt == '/history':
                        self._show_history()
                    elif prompt == '/clear':
                        self.generation_history.clear()
                        print("✅ Generation history cleared.")
                    elif prompt == '/quit':
                        print("👋 Goodbye!")
                        break
                    else:
                        print("❌ Unknown command. Type /help for available commands.")
                    continue
                
                # Generate text
                print("🔄 Generating...")
                start_time = time.time()
                
                generated_text = self.inference_engine.generate(
                    prompt=prompt,
                    **gen_params
                )
                
                generation_time = time.time() - start_time
                
                # Display result
                print(f"\n📝 Generated text ({generation_time:.2f}s):")
                print("-" * 40)
                print(generated_text)
                print("-" * 40)
                
                # Save to history
                self.generation_history.append({
                    'prompt': prompt,
                    'generated': generated_text,
                    'parameters': gen_params.copy(),
                    'generation_time': generation_time,
                    'timestamp': time.time()
                })
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during generation: {e}")
                logging.error(f"Generation error: {e}")
    
    def _show_help(self) -> None:
        """Show help message."""
        print("\n📖 Help:")
        print("Available parameters for /set command:")
        print("  max_length   - Maximum generation length (int)")
        print("  temperature  - Sampling temperature (float, 0.1-2.0)")
        print("  top_p        - Nucleus sampling threshold (float, 0.0-1.0)")
        print("  top_k        - Top-k sampling parameter (int, 0-100)")
        print("  do_sample    - Use sampling vs greedy (true/false)")
        print("\nExample: /set temperature 1.2")
    
    def _show_settings(self, gen_params: Dict[str, Any]) -> None:
        """Show current generation settings."""
        print("\n⚙️  Current settings:")
        for param, value in gen_params.items():
            print(f"  {param}: {value}")
    
    def _handle_set_command(self, command: str, gen_params: Dict[str, Any]) -> None:
        """Handle parameter setting command."""
        try:
            parts = command.split()
            if len(parts) != 3:
                print("❌ Usage: /set <parameter> <value>")
                return
            
            param_name = parts[1]
            param_value = parts[2]
            
            if param_name not in gen_params:
                print(f"❌ Unknown parameter: {param_name}")
                return
            
            # Convert value to appropriate type
            if param_name == 'do_sample':
                gen_params[param_name] = param_value.lower() in ['true', '1', 'yes']
            elif param_name in ['max_length', 'top_k']:
                gen_params[param_name] = int(param_value)
            else:
                gen_params[param_name] = float(param_value)
            
            print(f"✅ Set {param_name} = {gen_params[param_name]}")
            
        except ValueError as e:
            print(f"❌ Invalid value: {e}")
        except Exception as e:
            print(f"❌ Error setting parameter: {e}")
    
    def _show_history(self) -> None:
        """Show generation history."""
        if not self.generation_history:
            print("📝 No generation history.")
            return
        
        print(f"\n📚 Generation history ({len(self.generation_history)} entries):")
        for i, entry in enumerate(self.generation_history[-5:], 1):  # Show last 5
            print(f"\n{i}. Prompt: {entry['prompt'][:50]}...")
            print(f"   Generated: {entry['generated'][:100]}...")
            print(f"   Time: {entry['generation_time']:.2f}s")


class BatchGenerator:
    """Batch text generation from file inputs."""
    
    def __init__(self, inference_engine: MambaInference):
        """Initialize batch generator.
        
        Args:
            inference_engine: Configured MambaInference engine
        """
        self.inference_engine = inference_engine
    
    def process_batch_file(self, 
                          input_file: str,
                          output_file: str,
                          generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Process batch generation from input file.
        
        Args:
            input_file: Path to input file with prompts
            output_file: Path to output file for results
            generation_params: Generation parameters
            
        Returns:
            Dictionary with batch processing results
        """
        logging.info(f"Processing batch file: {input_file}")
        
        # Read input prompts
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Support different input formats
        prompts = []
        if input_path.suffix.lower() == '.json':
            with open(input_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = data
                elif isinstance(data, dict) and 'prompts' in data:
                    prompts = data['prompts']
                else:
                    raise ValueError("JSON file must contain list of prompts or dict with 'prompts' key")
        else:
            # Treat as text file with one prompt per line
            with open(input_path, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        
        if not prompts:
            raise ValueError("No prompts found in input file")
        
        logging.info(f"Found {len(prompts)} prompts to process")
        
        # Process each prompt
        results = []
        total_start_time = time.time()
        
        for i, prompt in enumerate(prompts, 1):
            logging.info(f"Processing prompt {i}/{len(prompts)}")
            
            start_time = time.time()
            try:
                generated_text = self.inference_engine.generate(
                    prompt=prompt,
                    **generation_params
                )
                generation_time = time.time() - start_time
                
                result = {
                    'prompt_id': i,
                    'prompt': prompt,
                    'generated': generated_text,
                    'generation_time': generation_time,
                    'success': True
                }
                
            except Exception as e:
                logging.error(f"Error processing prompt {i}: {e}")
                result = {
                    'prompt_id': i,
                    'prompt': prompt,
                    'error': str(e),
                    'generation_time': 0,
                    'success': False
                }
            
            results.append(result)
        
        total_time = time.time() - total_start_time
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        batch_results = {
            'input_file': input_file,
            'output_file': output_file,
            'generation_params': generation_params,
            'total_prompts': len(prompts),
            'successful_generations': sum(1 for r in results if r['success']),
            'total_time': total_time,
            'avg_time_per_prompt': total_time / len(prompts),
            'results': results,
            'timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        logging.info(f"Saved batch results to: {output_file}")
        logging.info(f"Processed {len(prompts)} prompts in {total_time:.2f}s")
        logging.info(f"Success rate: {batch_results['successful_generations']}/{len(prompts)}")
        
        return batch_results


def create_inference_engine(model_path: str, 
                          config_path: str,
                          device: Optional[torch.device] = None) -> MambaInference:
    """Create and initialize inference engine.
    
    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to model configuration
        device: Device for inference (auto-detect if None)
        
    Returns:
        Initialized MambaInference engine
    """
    # Load configuration
    config = ConfigLoader.load_from_file(config_path)
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Using device: {device}")
    
    # Load model
    model = load_mamba_model(model_path, config.model)
    model = model.to(device)
    model.eval()
    
    logging.info(f"Loaded model from: {model_path}")
    
    # Create inference engine
    inference_engine = MambaInference(
        model=model,
        tokenizer_path=config.data.tokenizer_path,
        config=config
    )
    
    return inference_engine


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Generate text using trained Mamba model")
    
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
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive text generation"
    )
    
    mode_group.add_argument(
        "--batch",
        type=str,
        help="Path to batch input file (JSON or text)"
    )
    
    mode_group.add_argument(
        "--single",
        type=str,
        help="Generate text from single prompt"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    
    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    
    # Batch mode options
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for batch mode results"
    )
    
    # Other options
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for inference"
    )
    
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
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Create inference engine
    try:
        inference_engine = create_inference_engine(
            model_path=args.model_path,
            config_path=args.config,
            device=device
        )
    except Exception as e:
        logging.error(f"Error creating inference engine: {e}")
        sys.exit(1)
    
    # Prepare generation parameters
    generation_params = {
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'do_sample': not args.no_sampling
    }
    
    # Run appropriate mode
    if args.interactive:
        # Interactive mode
        generator = InteractiveGenerator(inference_engine)
        generator.run_interactive_session()
        
    elif args.batch:
        # Batch mode
        if not args.output_file:
            # Generate default output filename
            input_path = Path(args.batch)
            args.output_file = str(input_path.parent / f"{input_path.stem}_results.json")
        
        generator = BatchGenerator(inference_engine)
        try:
            results = generator.process_batch_file(
                input_file=args.batch,
                output_file=args.output_file,
                generation_params=generation_params
            )
            
            print(f"✅ Batch processing completed!")
            print(f"📊 Results: {results['successful_generations']}/{results['total_prompts']} successful")
            print(f"⏱️  Total time: {results['total_time']:.2f}s")
            print(f"💾 Results saved to: {args.output_file}")
            
        except Exception as e:
            logging.error(f"Error in batch processing: {e}")
            sys.exit(1)
    
    elif args.single:
        # Single prompt mode
        try:
            print(f"🔤 Prompt: {args.single}")
            print("🔄 Generating...")
            
            start_time = time.time()
            generated_text = inference_engine.generate(
                prompt=args.single,
                **generation_params
            )
            generation_time = time.time() - start_time
            
            print(f"\n📝 Generated text ({generation_time:.2f}s):")
            print("-" * 60)
            print(generated_text)
            print("-" * 60)
            
        except Exception as e:
            logging.error(f"Error generating text: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()