"""
MambaInference class for efficient text generation with Mamba models.

This module implements autoregressive text generation with various sampling strategies
and state caching for efficient inference.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union, Tuple
import time
import logging
from dataclasses import dataclass
import threading
from collections import defaultdict
import psutil
import gc

from ..models.mamba_model import MambaModel
from ..config import MambaConfig


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_length: int = 100
    max_new_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    use_cache: bool = True
    
    def __post_init__(self):
        """Validate generation configuration."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


@dataclass
class GenerationOutput:
    """Output from text generation."""
    
    sequences: torch.Tensor
    scores: Optional[torch.Tensor] = None
    past_key_values: Optional[Dict[str, Any]] = None
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class BatchGenerationOutput:
    """Output from batch text generation."""
    
    sequences: List[torch.Tensor]
    generation_times: List[float]
    tokens_per_second: List[float]
    total_time: float = 0.0
    average_tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    batch_efficiency: float = 0.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for inference monitoring."""
    
    throughput_tokens_per_second: float = 0.0
    latency_ms_per_token: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cache_hit_rate: float = 0.0
    batch_efficiency: float = 0.0
    queue_length: int = 0


class MambaInference:
    """
    Efficient inference engine for Mamba models with autoregressive generation.
    
    This class provides text generation capabilities with various sampling strategies
    and optimizations for efficient inference including state caching.
    
    Args:
        model: Trained MambaModel for inference
        tokenizer: Tokenizer for encoding/decoding text (optional)
        config: Generation configuration
        device: Device to run inference on
    """
    
    def __init__(
        self,
        model: MambaModel,
        tokenizer=None,
        config: Optional[GenerationConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.device = device or next(model.parameters()).device
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Enhanced state caching system
        self._state_cache = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self._max_cache_size = 1000  # Maximum number of cached states
        
        # Performance tracking
        self._generation_stats = {
            'total_tokens_generated': 0,
            'total_generation_time': 0.0,
            'num_generations': 0,
            'batch_generations': 0,
            'total_batch_time': 0.0
        }
        
        # Batch processing configuration
        self._batch_config = {
            'max_batch_size': 32,
            'dynamic_batching': True,
            'padding_strategy': 'longest',  # 'longest' or 'fixed'
            'timeout_ms': 100  # Timeout for batch collection
        }
        
        # Performance monitoring
        self._performance_monitor = PerformanceMonitor()
        self._lock = threading.Lock()  # For thread-safe operations
    
    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> GenerationOutput:
        """
        Generate text using autoregressive decoding.
        
        Args:
            input_ids: Input token ids of shape (batch_size, seq_len)
            generation_config: Generation configuration (uses default if None)
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationOutput containing generated sequences and metadata
        """
        # Use provided config or default
        config = generation_config or self.config
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Validate inputs
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D tensor (batch_size, seq_len)")
        
        batch_size, input_length = input_ids.shape
        
        # Determine generation length
        if config.max_new_tokens is not None:
            max_length = input_length + config.max_new_tokens
        else:
            max_length = config.max_length
        
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        
        # Start timing
        start_time = time.time()
        
        # Generate sequences
        if config.do_sample:
            sequences = self._sample_generate(input_ids, config, max_length)
        else:
            sequences = self._greedy_generate(input_ids, config, max_length)
        
        # Calculate timing and performance metrics
        generation_time = time.time() - start_time
        total_new_tokens = (sequences.shape[1] - input_length) * batch_size
        tokens_per_second = total_new_tokens / generation_time if generation_time > 0 else 0
        
        # Update stats
        self._update_stats(total_new_tokens, generation_time)
        
        return GenerationOutput(
            sequences=sequences,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second
        )
    
    def _greedy_generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        max_length: int
    ) -> torch.Tensor:
        """Generate using greedy decoding (always pick most likely token)."""
        
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        
        # Initialize state cache if using caching
        if config.use_cache:
            cache = self._initialize_cache(batch_size, max_length)
        else:
            cache = None
        
        # Generate tokens one by one
        for step in range(input_ids.shape[1], max_length):
            with torch.no_grad():
                # Get logits for next token
                if config.use_cache and step > input_ids.shape[1]:
                    # Use cached states for efficiency
                    logits = self._forward_with_cache(
                        generated[:, -1:], cache, step
                    )
                else:
                    # Full forward pass
                    outputs = self.model(generated, return_dict=True)
                    logits = outputs.logits[:, -1, :]
                    
                    # Update cache if using caching
                    if config.use_cache:
                        self._update_cache(cache, outputs, step)
                
                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(
                        logits, generated, config.repetition_penalty
                    )
                
                # Greedy selection
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for end-of-sequence
                if (config.eos_token_id is not None and 
                    (next_token == config.eos_token_id).all()):
                    break
        
        return generated
    
    def _sample_generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        max_length: int
    ) -> torch.Tensor:
        """Generate using sampling with temperature, top-p, and top-k."""
        
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        
        # Initialize state cache if using caching
        if config.use_cache:
            cache = self._initialize_cache(batch_size, max_length)
        else:
            cache = None
        
        # Generate tokens one by one
        for step in range(input_ids.shape[1], max_length):
            with torch.no_grad():
                # Get logits for next token
                if config.use_cache and step > input_ids.shape[1]:
                    # Use cached states for efficiency
                    logits = self._forward_with_cache(
                        generated[:, -1:], cache, step
                    )
                else:
                    # Full forward pass
                    outputs = self.model(generated, return_dict=True)
                    logits = outputs.logits[:, -1, :]
                    
                    # Update cache if using caching
                    if config.use_cache:
                        self._update_cache(cache, outputs, step)
                
                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(
                        logits, generated, config.repetition_penalty
                    )
                
                # Apply temperature
                if config.temperature != 1.0:
                    logits = logits / config.temperature
                
                # Apply top-k filtering
                if config.top_k > 0:
                    logits = self._apply_top_k_filtering(logits, config.top_k)
                
                # Apply top-p (nucleus) filtering
                if config.top_p < 1.0:
                    logits = self._apply_top_p_filtering(logits, config.top_p)
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for end-of-sequence
                if (config.eos_token_id is not None and 
                    (next_token == config.eos_token_id).all()):
                    break
        
        return generated
    
    def _apply_top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        if top_k <= 0:
            return logits
        
        # Get top-k values and indices
        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        
        # Create filtered logits tensor
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(1, top_k_indices, top_k_logits)
        
        return logits_filtered
    
    def _apply_top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        if top_p >= 1.0:
            return logits
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift indices to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to reduce repetitive generation."""
        if penalty == 1.0:
            return logits
        
        # Clone logits to avoid modifying the original
        logits = logits.clone()
        
        # Get unique tokens in generated sequence for each batch
        for batch_idx in range(generated.shape[0]):
            unique_tokens = generated[batch_idx].unique()
            for token in unique_tokens:
                # Apply penalty (reduce probability if penalty > 1.0)
                if logits[batch_idx, token] > 0:
                    logits[batch_idx, token] /= penalty
                else:
                    logits[batch_idx, token] *= penalty
        
        return logits
    
    def _initialize_cache(self, batch_size: int, max_length: int) -> Dict[str, Any]:
        """Initialize state cache for efficient generation."""
        # For now, return empty cache - will be implemented with actual state caching
        # when the Mamba model supports it
        return {}
    
    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache: Dict[str, Any],
        step: int
    ) -> torch.Tensor:
        """Forward pass using cached states (placeholder for now)."""
        # For now, fall back to full forward pass
        # This will be optimized when proper state caching is implemented
        outputs = self.model(input_ids, return_dict=True)
        return outputs.logits[:, -1, :]
    
    def _update_cache(
        self,
        cache: Dict[str, Any],
        outputs: Any,
        step: int
    ) -> None:
        """Update state cache with current outputs (placeholder for now)."""
        # Placeholder for cache update logic
        pass
    
    def generate_text(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a string prompt.
        
        Args:
            prompt: Input text prompt
            generation_config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text as string
            
        Raises:
            ValueError: If no tokenizer is provided
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for text generation")
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate
        output = self.generate(input_ids, generation_config, **kwargs)
        
        # Decode generated sequence
        generated_text = self.tokenizer.decode(
            output.sequences[0], skip_special_tokens=True
        )
        
        return generated_text
    
    def batch_generate_text(
        self,
        prompts: List[str],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text from multiple prompts in batch.
        
        Args:
            prompts: List of input text prompts
            generation_config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
            
        Raises:
            ValueError: If no tokenizer is provided
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for text generation")
        
        # Encode prompts with padding
        encoded = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        input_ids = encoded['input_ids']
        
        # Generate
        output = self.generate(input_ids, generation_config, **kwargs)
        
        # Decode generated sequences
        generated_texts = []
        for sequence in output.sequences:
            text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def _update_stats(self, tokens_generated: int, generation_time: float) -> None:
        """Update generation statistics."""
        self._generation_stats['total_tokens_generated'] += tokens_generated
        self._generation_stats['total_generation_time'] += generation_time
        self._generation_stats['num_generations'] += 1
    
    def get_generation_stats(self) -> Dict[str, float]:
        """Get generation performance statistics."""
        stats = self._generation_stats.copy()
        
        if stats['total_generation_time'] > 0:
            stats['average_tokens_per_second'] = (
                stats['total_tokens_generated'] / stats['total_generation_time']
            )
        else:
            stats['average_tokens_per_second'] = 0.0
        
        if stats['num_generations'] > 0:
            stats['average_tokens_per_generation'] = (
                stats['total_tokens_generated'] / stats['num_generations']
            )
            stats['average_time_per_generation'] = (
                stats['total_generation_time'] / stats['num_generations']
            )
        else:
            stats['average_tokens_per_generation'] = 0.0
            stats['average_time_per_generation'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset generation statistics."""
        self._generation_stats = {
            'total_tokens_generated': 0,
            'total_generation_time': 0.0,
            'num_generations': 0,
            'batch_generations': 0,
            'total_batch_time': 0.0
        }
    
    def clear_cache(self) -> None:
        """Clear the state cache."""
        self._state_cache.clear()
    
    def set_generation_config(self, config: GenerationConfig) -> None:
        """Set default generation configuration."""
        self.config = config
    
    def batch_generate(
        self,
        input_ids_list: List[torch.Tensor],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> BatchGenerationOutput:
        """
        Generate text for multiple inputs with optimized batching.
        
        Args:
            input_ids_list: List of input token tensors
            generation_config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            BatchGenerationOutput with results for each input
        """
        config = generation_config or self.config
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        start_time = time.time()
        
        # Optimize batch processing
        if self._batch_config['dynamic_batching']:
            results = self._dynamic_batch_generate(input_ids_list, config)
        else:
            results = self._static_batch_generate(input_ids_list, config)
        
        total_time = time.time() - start_time
        
        # Calculate batch efficiency metrics
        total_tokens = sum(seq.numel() for seq in results.sequences)
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        
        # Estimate batch efficiency (vs individual generation)
        estimated_individual_time = len(input_ids_list) * (total_time / len(input_ids_list))
        batch_efficiency = estimated_individual_time / total_time if total_time > 0 else 1.0
        
        results.total_time = total_time
        results.average_tokens_per_second = avg_tps
        results.batch_efficiency = batch_efficiency
        results.memory_usage_mb = self._get_memory_usage()
        
        # Update stats
        with self._lock:
            self._generation_stats['batch_generations'] += 1
            self._generation_stats['total_batch_time'] += total_time
        
        return results
    
    def _dynamic_batch_generate(
        self,
        input_ids_list: List[torch.Tensor],
        config: GenerationConfig
    ) -> BatchGenerationOutput:
        """Generate with dynamic batching for optimal memory usage."""
        
        # Sort by sequence length for better batching
        sorted_inputs = sorted(
            enumerate(input_ids_list),
            key=lambda x: x[1].shape[1]
        )
        
        results = BatchGenerationOutput(
            sequences=[None] * len(input_ids_list),
            generation_times=[0.0] * len(input_ids_list),
            tokens_per_second=[0.0] * len(input_ids_list)
        )
        
        # Process in batches
        max_batch_size = self._batch_config['max_batch_size']
        
        for i in range(0, len(sorted_inputs), max_batch_size):
            batch_items = sorted_inputs[i:i + max_batch_size]
            batch_indices = [item[0] for item in batch_items]
            batch_inputs = [item[1] for item in batch_items]
            
            # Create padded batch
            batch_tensor, attention_mask = self._create_padded_batch(batch_inputs)
            
            # Generate for batch
            batch_start = time.time()
            batch_output = self.generate(batch_tensor, config)
            batch_time = time.time() - batch_start
            
            # Distribute results back to original positions
            for j, (orig_idx, input_tensor) in enumerate(batch_items):
                seq_len = input_tensor.shape[1]
                generated_seq = batch_output.sequences[j]
                
                # Remove padding if necessary
                if attention_mask is not None:
                    # Find actual sequence length
                    actual_len = attention_mask[j].sum().item()
                    generated_seq = generated_seq[:actual_len]
                
                results.sequences[orig_idx] = generated_seq
                results.generation_times[orig_idx] = batch_time / len(batch_items)
                
                new_tokens = generated_seq.shape[0] - seq_len
                results.tokens_per_second[orig_idx] = (
                    new_tokens / results.generation_times[orig_idx]
                    if results.generation_times[orig_idx] > 0 else 0
                )
        
        return results
    
    def _static_batch_generate(
        self,
        input_ids_list: List[torch.Tensor],
        config: GenerationConfig
    ) -> BatchGenerationOutput:
        """Generate with static batching (simpler but less efficient)."""
        
        results = BatchGenerationOutput(
            sequences=[],
            generation_times=[],
            tokens_per_second=[]
        )
        
        # Process all inputs in a single batch
        batch_tensor, attention_mask = self._create_padded_batch(input_ids_list)
        
        start_time = time.time()
        batch_output = self.generate(batch_tensor, config)
        total_time = time.time() - start_time
        
        # Distribute results
        for i, input_tensor in enumerate(input_ids_list):
            seq_len = input_tensor.shape[1]
            generated_seq = batch_output.sequences[i]
            
            # Remove padding if necessary
            if attention_mask is not None:
                actual_len = attention_mask[i].sum().item()
                generated_seq = generated_seq[:actual_len]
            
            results.sequences.append(generated_seq)
            results.generation_times.append(total_time / len(input_ids_list))
            
            new_tokens = generated_seq.shape[0] - seq_len
            results.tokens_per_second.append(
                new_tokens / results.generation_times[-1]
                if results.generation_times[-1] > 0 else 0
            )
        
        return results
    
    def _create_padded_batch(
        self,
        input_ids_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Create a padded batch tensor from list of input tensors."""
        
        if not input_ids_list:
            raise ValueError("Empty input list")
        
        # Find maximum sequence length
        max_len = max(tensor.shape[1] for tensor in input_ids_list)
        batch_size = len(input_ids_list)
        
        # Create padded batch tensor
        pad_token_id = self.config.pad_token_id or 0
        batch_tensor = torch.full(
            (batch_size, max_len),
            pad_token_id,
            dtype=input_ids_list[0].dtype,
            device=self.device
        )
        
        # Create attention mask
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
            device=self.device
        )
        
        # Fill batch tensor and attention mask
        for i, tensor in enumerate(input_ids_list):
            seq_len = tensor.shape[1]
            batch_tensor[i, :seq_len] = tensor.squeeze(0)
            attention_mask[i, :seq_len] = True
        
        return batch_tensor, attention_mask
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        else:
            # For CPU, use process memory
            process = psutil.Process()
            return process.memory_info().rss / (1024 ** 2)
    
    def optimize_cache(self, max_size: Optional[int] = None) -> None:
        """Optimize state cache by removing least recently used entries."""
        if max_size is not None:
            self._max_cache_size = max_size
        
        if len(self._state_cache) > self._max_cache_size:
            # Simple LRU eviction (remove oldest entries)
            items_to_remove = len(self._state_cache) - self._max_cache_size
            keys_to_remove = list(self._state_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self._state_cache[key]
                self._cache_stats['evictions'] += 1
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        stats = self._generation_stats
        cache_stats = self._cache_stats
        
        # Calculate cache hit rate
        total_cache_requests = cache_stats['hits'] + cache_stats['misses']
        cache_hit_rate = (
            cache_stats['hits'] / total_cache_requests
            if total_cache_requests > 0 else 0.0
        )
        
        # Calculate average throughput
        total_time = stats['total_generation_time'] + stats['total_batch_time']
        avg_throughput = (
            stats['total_tokens_generated'] / total_time
            if total_time > 0 else 0.0
        )
        
        # Calculate average latency per token
        total_generations = stats['num_generations'] + stats['batch_generations']
        avg_latency = (
            (total_time * 1000) / stats['total_tokens_generated']
            if stats['total_tokens_generated'] > 0 else 0.0
        )
        
        # Calculate batch efficiency
        if stats['batch_generations'] > 0:
            avg_batch_time = stats['total_batch_time'] / stats['batch_generations']
            avg_single_time = (
                stats['total_generation_time'] / stats['num_generations']
                if stats['num_generations'] > 0 else avg_batch_time
            )
            batch_efficiency = avg_single_time / avg_batch_time if avg_batch_time > 0 else 1.0
        else:
            batch_efficiency = 0.0
        
        return PerformanceMetrics(
            throughput_tokens_per_second=avg_throughput,
            latency_ms_per_token=avg_latency,
            memory_usage_mb=self._get_memory_usage(),
            cache_hit_rate=cache_hit_rate,
            batch_efficiency=batch_efficiency,
            queue_length=len(self._state_cache)
        )
    
    def set_batch_config(self, **kwargs) -> None:
        """Update batch processing configuration."""
        for key, value in kwargs.items():
            if key in self._batch_config:
                self._batch_config[key] = value
            else:
                raise ValueError(f"Unknown batch config parameter: {key}")
    
    def get_batch_config(self) -> Dict[str, Any]:
        """Get current batch processing configuration."""
        return self._batch_config.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        param_counts = self.model.count_parameters()
        memory_footprint = self.model.get_memory_footprint()
        
        return {
            'model_config': self.model.config,
            'parameter_counts': param_counts,
            'memory_footprint_mb': memory_footprint,
            'device': str(self.device),
            'model_dtype': str(next(self.model.parameters()).dtype),
            'cache_size': len(self._state_cache),
            'max_cache_size': self._max_cache_size
        }


class PerformanceMonitor:
    """Monitor and track inference performance metrics."""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = time.time()
        
    def record_generation(
        self,
        tokens_generated: int,
        generation_time: float,
        memory_usage: float
    ) -> None:
        """Record metrics for a generation event."""
        timestamp = time.time()
        
        metrics = {
            'timestamp': timestamp,
            'tokens_generated': tokens_generated,
            'generation_time': generation_time,
            'memory_usage_mb': memory_usage,
            'tokens_per_second': tokens_generated / generation_time if generation_time > 0 else 0
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only recent history (last 1000 entries)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_average_metrics(self, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get average metrics over a time window."""
        if not self.metrics_history:
            return {}
        
        # Filter by time window if specified
        if window_seconds is not None:
            cutoff_time = time.time() - window_seconds
            recent_metrics = [
                m for m in self.metrics_history 
                if m['timestamp'] >= cutoff_time
            ]
        else:
            recent_metrics = self.metrics_history
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        total_tokens = sum(m['tokens_generated'] for m in recent_metrics)
        total_time = sum(m['generation_time'] for m in recent_metrics)
        avg_memory = sum(m['memory_usage_mb'] for m in recent_metrics) / len(recent_metrics)
        
        return {
            'average_tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            'average_memory_usage_mb': avg_memory,
            'total_generations': len(recent_metrics),
            'total_tokens': total_tokens,
            'total_time': total_time
        }


def create_inference_engine(
    model: MambaModel,
    tokenizer=None,
    generation_config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None
) -> MambaInference:
    """
    Factory function to create a MambaInference engine.
    
    Args:
        model: Trained MambaModel
        tokenizer: Tokenizer for text encoding/decoding
        generation_config: Default generation configuration
        device: Device for inference
        
    Returns:
        Configured MambaInference engine
    """
    return MambaInference(
        model=model,
        tokenizer=tokenizer,
        config=generation_config,
        device=device
    )


def create_optimized_inference_engine(
    model: MambaModel,
    tokenizer=None,
    generation_config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
    max_batch_size: int = 32,
    max_cache_size: int = 1000
) -> MambaInference:
    """
    Factory function to create an optimized MambaInference engine.
    
    Args:
        model: Trained MambaModel
        tokenizer: Tokenizer for text encoding/decoding
        generation_config: Default generation configuration
        device: Device for inference
        max_batch_size: Maximum batch size for batch processing
        max_cache_size: Maximum size of state cache
        
    Returns:
        Optimized MambaInference engine
    """
    engine = MambaInference(
        model=model,
        tokenizer=tokenizer,
        config=generation_config,
        device=device
    )
    
    # Configure for optimal performance
    engine.set_batch_config(
        max_batch_size=max_batch_size,
        dynamic_batching=True,
        padding_strategy='longest'
    )
    
    engine._max_cache_size = max_cache_size
    
    return engine