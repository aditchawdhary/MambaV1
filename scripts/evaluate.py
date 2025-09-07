#!/usr/bin/env python3
"""
Evaluation script for Mamba models on test datasets.

This script provides comprehensive evaluation capabilities including:
- Model validation on test sets
- Perplexity and loss computation
- Generation quality assessment
- Performance benchmarking
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mamba_training.config import Config, ConfigLoader
from mamba_training.models.mamba_model import MambaModel, load_mamba_model
from mamba_training.data.dataset_processor import DatasetProcessor
from mamba_training.data.data_loader import MambaDataLoader
from mamba_training.inference.mamba_inference import MambaInference


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model: MambaModel, config: Config, device: torch.device):
        """Initialize evaluator.
        
        Args:
            model: Trained Mamba model
            config: Model configuration
            device: Device for evaluation
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()
        
        # Initialize inference engine
        self.inference_engine = MambaInference(
            model=model,
            tokenizer_path=config.data.tokenizer_path,
            config=config
        )
    
    def evaluate_perplexity(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model perplexity on dataset.
        
        Args:
            dataloader: DataLoader for evaluation dataset
            
        Returns:
            Dictionary containing perplexity metrics
        """
        logging.info("Evaluating perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Extract loss
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    # Calculate loss manually
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    labels = batch.get('labels', batch.get('input_ids'))
                    
                    # Shift labels for next token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                
                # Accumulate metrics
                batch_tokens = (batch.get('labels', batch.get('input_ids')) != -100).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                num_batches += 1
        
        # Calculate final metrics
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens,
            'num_batches': num_batches
        }
    
    def evaluate_generation_quality(self, 
                                  prompts: List[str],
                                  max_length: int = 100,
                                  num_samples: int = 5) -> Dict[str, Any]:
        """Evaluate text generation quality.
        
        Args:
            prompts: List of prompts for generation
            max_length: Maximum generation length
            num_samples: Number of samples per prompt
            
        Returns:
            Dictionary containing generation metrics
        """
        logging.info("Evaluating generation quality...")
        
        generation_results = []
        generation_times = []
        
        for prompt in prompts:
            prompt_results = []
            
            for _ in range(num_samples):
                start_time = time.time()
                
                # Generate text
                generated_text = self.inference_engine.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True
                )
                
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                prompt_results.append({
                    'prompt': prompt,
                    'generated': generated_text,
                    'generation_time': generation_time,
                    'length': len(generated_text.split())
                })
            
            generation_results.extend(prompt_results)
        
        # Calculate metrics
        avg_generation_time = np.mean(generation_times)
        avg_length = np.mean([r['length'] for r in generation_results])
        
        return {
            'generations': generation_results,
            'avg_generation_time': avg_generation_time,
            'avg_length': avg_length,
            'total_samples': len(generation_results)
        }
    
    def benchmark_inference_speed(self, 
                                batch_sizes: List[int] = [1, 4, 8, 16],
                                sequence_lengths: List[int] = [128, 256, 512]) -> Dict[str, Any]:
        """Benchmark inference speed across different configurations.
        
        Args:
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            
        Returns:
            Dictionary containing benchmark results
        """
        logging.info("Benchmarking inference speed...")
        
        benchmark_results = []
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                # Create dummy input
                input_ids = torch.randint(
                    0, self.config.model.vocab_size,
                    (batch_size, seq_len),
                    device=self.device
                )
                
                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = self.model(input_ids)
                
                # Benchmark
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                num_runs = 20
                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = self.model(input_ids)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                avg_time_per_run = total_time / num_runs
                tokens_per_second = (batch_size * seq_len) / avg_time_per_run
                
                benchmark_results.append({
                    'batch_size': batch_size,
                    'sequence_length': seq_len,
                    'avg_time_per_run': avg_time_per_run,
                    'tokens_per_second': tokens_per_second,
                    'total_tokens': batch_size * seq_len
                })
        
        return {
            'benchmark_results': benchmark_results,
            'device': str(self.device),
            'model_parameters': self.model.count_parameters()['total']
        }
    
    def evaluate_memory_usage(self) -> Dict[str, float]:
        """Evaluate model memory usage."""
        logging.info("Evaluating memory usage...")
        
        memory_stats = {}
        
        if torch.cuda.is_available():
            # GPU memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Forward pass to measure memory
            dummy_input = torch.randint(
                0, self.config.model.vocab_size,
                (1, 512),
                device=self.device
            )
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            memory_stats.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                'gpu_max_memory_allocated_mb': torch.cuda.max_memory_allocated() / (1024**2),
            })
        
        # Model parameter memory
        model_memory = self.model.get_memory_footprint()
        memory_stats.update(model_memory)
        
        return memory_stats


def create_evaluation_dataset(config: Config, dataset_path: str) -> DataLoader:
    """Create evaluation dataset and dataloader.
    
    Args:
        config: Model configuration
        dataset_path: Path to evaluation dataset
        
    Returns:
        DataLoader for evaluation
    """
    # Initialize dataset processor
    processor = DatasetProcessor(
        tokenizer=config.data.tokenizer_path,
        config=config.data
    )
    
    # Process dataset
    dataset = processor.process_dataset(dataset_path)
    
    # Create dataloader
    dataloader = MambaDataLoader(
        dataset=dataset,
        config=config.data,
        batch_size=config.training.batch_size,
        shuffle=False,
        use_sequence_packing=False,
        distributed=False
    )
    
    return dataloader


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Mamba model performance")
    
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
    
    # Dataset arguments
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test dataset"
    )
    
    # Evaluation options
    parser.add_argument(
        "--eval-perplexity",
        action="store_true",
        help="Evaluate model perplexity"
    )
    
    parser.add_argument(
        "--eval-generation",
        action="store_true",
        help="Evaluate text generation quality"
    )
    
    parser.add_argument(
        "--benchmark-speed",
        action="store_true",
        help="Benchmark inference speed"
    )
    
    parser.add_argument(
        "--eval-memory",
        action="store_true",
        help="Evaluate memory usage"
    )
    
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Run all evaluation metrics"
    )
    
    # Generation options
    parser.add_argument(
        "--generation-prompts",
        type=str,
        nargs="+",
        default=["The quick brown fox", "In a world where", "Once upon a time"],
        help="Prompts for generation evaluation"
    )
    
    parser.add_argument(
        "--max-generation-length",
        type=int,
        default=100,
        help="Maximum length for text generation"
    )
    
    parser.add_argument(
        "--num-generation-samples",
        type=int,
        default=3,
        help="Number of samples per prompt"
    )
    
    # Output options
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save evaluation results"
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
    
    # Load configuration
    try:
        config = ConfigLoader.load_from_file(args.config)
        logging.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model
    try:
        model = load_mamba_model(args.model_path, config.model)
        logging.info(f"Loaded model from {args.model_path}")
        
        # Log model info
        param_counts = model.count_parameters()
        logging.info(f"Model parameters: {param_counts}")
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, config, device)
    
    # Determine which evaluations to run
    run_perplexity = args.eval_perplexity or args.eval_all
    run_generation = args.eval_generation or args.eval_all
    run_benchmark = args.benchmark_speed or args.eval_all
    run_memory = args.eval_memory or args.eval_all
    
    # Store all results
    results = {
        'model_path': args.model_path,
        'config_path': args.config,
        'test_data_path': args.test_data,
        'device': str(device),
        'evaluation_timestamp': time.time()
    }
    
    # Evaluate perplexity
    if run_perplexity:
        try:
            logging.info("Creating evaluation dataset...")
            eval_dataloader = create_evaluation_dataset(config, args.test_data)
            
            perplexity_results = evaluator.evaluate_perplexity(eval_dataloader)
            results['perplexity'] = perplexity_results
            
            logging.info(f"Perplexity: {perplexity_results['perplexity']:.4f}")
            logging.info(f"Loss: {perplexity_results['loss']:.4f}")
            
        except Exception as e:
            logging.error(f"Error evaluating perplexity: {e}")
            results['perplexity_error'] = str(e)
    
    # Evaluate generation quality
    if run_generation:
        try:
            generation_results = evaluator.evaluate_generation_quality(
                prompts=args.generation_prompts,
                max_length=args.max_generation_length,
                num_samples=args.num_generation_samples
            )
            results['generation'] = generation_results
            
            logging.info(f"Average generation time: {generation_results['avg_generation_time']:.4f}s")
            logging.info(f"Average generation length: {generation_results['avg_length']:.1f} words")
            
            # Log sample generations
            for i, gen in enumerate(generation_results['generations'][:3]):
                logging.info(f"Sample {i+1}: {gen['prompt']} -> {gen['generated'][:100]}...")
                
        except Exception as e:
            logging.error(f"Error evaluating generation: {e}")
            results['generation_error'] = str(e)
    
    # Benchmark inference speed
    if run_benchmark:
        try:
            benchmark_results = evaluator.benchmark_inference_speed()
            results['benchmark'] = benchmark_results
            
            # Log key benchmark results
            for result in benchmark_results['benchmark_results']:
                logging.info(
                    f"Batch size {result['batch_size']}, "
                    f"Seq len {result['sequence_length']}: "
                    f"{result['tokens_per_second']:.0f} tokens/sec"
                )
                
        except Exception as e:
            logging.error(f"Error benchmarking speed: {e}")
            results['benchmark_error'] = str(e)
    
    # Evaluate memory usage
    if run_memory:
        try:
            memory_results = evaluator.evaluate_memory_usage()
            results['memory'] = memory_results
            
            logging.info(f"Model memory footprint: {memory_results}")
            
        except Exception as e:
            logging.error(f"Error evaluating memory: {e}")
            results['memory_error'] = str(e)
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logging.info(f"Saved evaluation results to {output_path}")
    
    # Print summary
    logging.info("Evaluation completed!")
    if 'perplexity' in results:
        logging.info(f"Final perplexity: {results['perplexity']['perplexity']:.4f}")
    
    return results


if __name__ == "__main__":
    main()