#!/usr/bin/env python3
"""
Demonstration script for the complete MambaModel architecture.

This script shows how to:
1. Create and configure a MambaModel
2. Perform forward passes
3. Generate text
4. Count parameters and estimate memory usage
5. Save and load models
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from mamba_training.config import MambaConfig
from mamba_training.models.mamba_model import MambaModel, create_mamba_model


def main():
    """Main demonstration function."""
    
    print("🐍 Mamba Model Architecture Demo")
    print("=" * 50)
    
    # 1. Create model configuration
    print("\n1. Creating Model Configuration")
    config = MambaConfig(
        d_model=256,        # Model dimension
        d_state=16,         # State space dimension
        d_conv=4,           # Convolution kernel size
        expand=2,           # Expansion factor
        n_layers=6,         # Number of Mamba layers
        vocab_size=10000,   # Vocabulary size
        pad_token_id=0      # Padding token ID
    )
    
    print(f"   Model dimension: {config.d_model}")
    print(f"   State dimension: {config.d_state}")
    print(f"   Number of layers: {config.n_layers}")
    print(f"   Vocabulary size: {config.vocab_size}")
    
    # 2. Create the model
    print("\n2. Creating MambaModel")
    model = create_mamba_model(config)
    
    # Count parameters
    param_counts = model.count_parameters()
    print(f"   Total parameters: {param_counts['total']:,}")
    print(f"   Trainable parameters: {param_counts['trainable']:,}")
    print(f"   Embedding parameters: {param_counts['embeddings']:,}")
    print(f"   MambaBlock parameters: {param_counts['mamba_blocks']:,}")
    
    # Memory footprint
    memory_info = model.get_memory_footprint()
    print(f"   Model memory: {memory_info['total_params_mb']:.2f} MB")
    
    # 3. Forward pass demonstration
    print("\n3. Forward Pass Demonstration")
    batch_size = 2
    seq_len = 20
    
    # Create sample input (random token IDs, avoiding padding token 0)
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    print(f"   Input shape: {input_ids.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
        logits = outputs.logits
        
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    print(f"   Probability range: [{probs.min():.6f}, {probs.max():.6f}]")
    
    # 4. Text generation demonstration
    print("\n4. Text Generation Demonstration")
    
    # Create a shorter prompt
    prompt_length = 5
    prompt_ids = torch.randint(1, config.vocab_size, (1, prompt_length))
    print(f"   Prompt shape: {prompt_ids.shape}")
    print(f"   Prompt tokens: {prompt_ids.squeeze().tolist()}")
    
    # Generate text with different strategies
    strategies = [
        {"name": "Greedy", "do_sample": False},
        {"name": "Temperature=0.8", "do_sample": True, "temperature": 0.8},
        {"name": "Top-p=0.9", "do_sample": True, "top_p": 0.9},
        {"name": "Top-k=50", "do_sample": True, "top_k": 50},
    ]
    
    for strategy in strategies:
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                max_length=15,
                **{k: v for k, v in strategy.items() if k != "name"}
            )
        
        generated_tokens = generated.squeeze().tolist()
        new_tokens = generated_tokens[prompt_length:]
        
        print(f"   {strategy['name']:15}: {new_tokens}")
    
    # 5. Hidden states analysis
    print("\n5. Hidden States Analysis")
    
    with torch.no_grad():
        outputs = model(
            input_ids[:1, :10],  # Use smaller input for analysis
            output_hidden_states=True,
            return_dict=True
        )
    
    hidden_states = outputs.hidden_states
    print(f"   Number of hidden state layers: {len(hidden_states)}")
    
    for i, hidden_state in enumerate(hidden_states):
        mean_activation = hidden_state.mean().item()
        std_activation = hidden_state.std().item()
        print(f"   Layer {i}: mean={mean_activation:.4f}, std={std_activation:.4f}")
    
    # 6. Model saving and loading
    print("\n6. Model Saving and Loading")
    
    # Save model
    save_path = "mamba_model_demo.pt"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'param_counts': param_counts,
        'memory_info': memory_info
    }
    
    torch.save(checkpoint, save_path)
    print(f"   Model saved to: {save_path}")
    
    # Load model
    from mamba_training.models.mamba_model import load_mamba_model
    loaded_model = load_mamba_model(save_path)
    print(f"   Model loaded successfully")
    
    # Verify loaded model produces same output
    with torch.no_grad():
        original_output = model(input_ids[:1, :5], return_dict=True)
        loaded_output = loaded_model(input_ids[:1, :5], return_dict=True)
        
        max_diff = (original_output.logits - loaded_output.logits).abs().max().item()
        print(f"   Max difference between original and loaded: {max_diff:.2e}")
    
    # Clean up
    Path(save_path).unlink()
    print(f"   Cleaned up saved file")
    
    # 7. Performance analysis
    print("\n7. Performance Analysis")
    
    # Measure inference time
    import time
    
    model.eval()
    warmup_runs = 5
    test_runs = 20
    
    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_ids)
    
    # Measure
    start_time = time.time()
    for _ in range(test_runs):
        with torch.no_grad():
            _ = model(input_ids)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / test_runs
    tokens_per_second = (batch_size * seq_len) / avg_time
    
    print(f"   Average inference time: {avg_time*1000:.2f} ms")
    print(f"   Tokens per second: {tokens_per_second:.0f}")
    
    # 8. Architecture summary
    print("\n8. Architecture Summary")
    print(f"   Model type: Mamba (State Space Model)")
    print(f"   Architecture: {config.n_layers} layers × {config.d_model} dimensions")
    print(f"   Parameters: {param_counts['total']:,} ({memory_info['total_params_mb']:.1f} MB)")
    print(f"   Context length: Unlimited (state space)")
    print(f"   Computational complexity: O(L) where L is sequence length")
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()