#!/usr/bin/env python3
"""
Demo script showing the complete data processing pipeline for Mamba training.

This script demonstrates:
1. Dataset processing and tokenization
2. Data loading with batching and sequence packing
3. Memory usage estimation
"""

import tempfile
import json
from pathlib import Path

from mamba_training.config import DataConfig
from mamba_training.data import (
    DatasetProcessor,
    TextQualityFilter,
    MambaDataLoader,
    create_data_loaders
)


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the way we process information.",
        "State space models like Mamba offer efficient sequence modeling.",
        "Natural language processing has made significant advances recently.",
        "Deep learning architectures continue to evolve and improve.",
        "Transformers revolutionized the field of artificial intelligence.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Large language models demonstrate impressive capabilities.",
        "Training neural networks requires careful optimization strategies.",
        "Data preprocessing is crucial for model performance."
    ]
    
    # Create temporary dataset file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(sample_texts, temp_file)
    temp_file.close()
    
    return temp_file.name


def main():
    """Run the data pipeline demonstration."""
    print("🚀 Mamba Data Pipeline Demo")
    print("=" * 50)
    
    # 1. Create configuration
    config = DataConfig(
        max_seq_length=64,
        preprocessing_batch_size=4,
        num_workers=0  # Use 0 for demo to avoid multiprocessing
    )
    
    print(f"📋 Configuration:")
    print(f"   Max sequence length: {config.max_seq_length}")
    print(f"   Batch size: {config.preprocessing_batch_size}")
    print()
    
    # 2. Create sample dataset
    dataset_path = create_sample_dataset()
    print(f"📁 Created sample dataset: {dataset_path}")
    
    try:
        # 3. Initialize dataset processor with quality filter
        quality_filter = TextQualityFilter(
            min_length=10,
            max_length=1000,
            min_word_count=3
        )
        
        # Create a simple mock tokenizer for demo
        class MockTokenizer:
            def __init__(self):
                self.pad_token = '<pad>'
                self.eos_token = '</s>'
                self.pad_token_id = 0
                self.eos_token_id = 1
            
            def __call__(self, text, **kwargs):
                # Simple word-based tokenization
                words = text.split()
                input_ids = [i + 2 for i in range(len(words))]
                max_length = kwargs.get('max_length', len(input_ids))
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                return {
                    'input_ids': input_ids,
                    'attention_mask': [1] * len(input_ids)
                }
            
            def __len__(self):
                return 10000
        
        tokenizer = MockTokenizer()
        processor = DatasetProcessor(tokenizer, config, quality_filter)
        
        print(f"🔧 Initialized DatasetProcessor")
        print(f"   Tokenizer vocab size: {processor.get_vocab_size()}")
        print(f"   Special tokens: {processor.get_special_token_ids()}")
        print()
        
        # 4. Process the dataset
        print("⚙️  Processing dataset...")
        processed_samples = processor.process_dataset(dataset_path)
        
        print(f"✅ Dataset processing complete!")
        print(f"   Processed {len(processed_samples)} samples")
        print()
        
        # 5. Create data loaders
        print("🔄 Creating data loaders...")
        
        # Split into train/val (simple split for demo)
        split_idx = int(len(processed_samples) * 0.8)
        train_samples = processed_samples[:split_idx]
        val_samples = processed_samples[split_idx:]
        
        from mamba_training.data.dataset_processor import ProcessedDataset
        train_dataset = ProcessedDataset(train_samples)
        val_dataset = ProcessedDataset(val_samples)
        
        train_loader, val_loader = create_data_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            distributed=False
        )
        
        print(f"✅ Data loaders created!")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print()
        
        # 6. Demonstrate memory usage estimation
        print("💾 Memory usage estimation:")
        memory_stats = train_loader.get_memory_usage()
        for key, value in memory_stats.items():
            print(f"   {key}: {value:.2f}")
        print()
        
        # 7. Iterate through a few batches
        print("🔍 Sample batch inspection:")
        for i, batch in enumerate(train_loader):
            if i >= 2:  # Only show first 2 batches
                break
            
            print(f"   Batch {i + 1}:")
            print(f"     Input IDs shape: {batch.input_ids.shape}")
            print(f"     Attention mask shape: {batch.attention_mask.shape}")
            print(f"     Labels shape: {batch.labels.shape if batch.labels is not None else 'None'}")
            print(f"     Sequence lengths: {batch.sequence_lengths.tolist()}")
            print()
        
        print("🎉 Demo completed successfully!")
        
    finally:
        # Clean up temporary file
        Path(dataset_path).unlink()
        print(f"🧹 Cleaned up temporary files")


if __name__ == "__main__":
    main()