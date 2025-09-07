#!/usr/bin/env python3
"""Download WikiText-103 dataset from HuggingFace."""

import os
from pathlib import Path
from datasets import load_dataset

def download_wikitext():
    """Download WikiText-103 from HuggingFace and save as text files."""
    
    print("🚀 Downloading WikiText-103 from HuggingFace...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download the dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    print(f"✅ Downloaded WikiText-103")
    print(f"   - Train samples: {len(dataset['train'])}")
    print(f"   - Validation samples: {len(dataset['validation'])}")
    print(f"   - Test samples: {len(dataset['test'])}")
    
    # Save train split as text file
    train_file = data_dir / "train.txt"
    print(f"💾 Saving training data to {train_file}...")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset['train']):
            text = example['text'].strip()
            if text:  # Skip empty lines
                f.write(text + '\n\n')
            
            if (i + 1) % 10000 == 0:
                print(f"   Processed {i + 1} samples...")
    
    # Save validation split
    val_file = data_dir / "validation.txt"
    print(f"💾 Saving validation data to {val_file}...")
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in dataset['validation']:
            text = example['text'].strip()
            if text:
                f.write(text + '\n\n')
    
    # Update config to use the downloaded data
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Update dataset path
        updated_config = config_content.replace(
            'dataset_path: "data/"',
            'dataset_path: "data/train.txt"'
        )
        
        with open(config_path, 'w') as f:
            f.write(updated_config)
        
        print(f"✅ Updated config to use WikiText-103 data")
    
    # Print file sizes
    train_size = train_file.stat().st_size / (1024 * 1024)  # MB
    val_size = val_file.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n📊 Dataset ready:")
    print(f"   - Training data: {train_size:.1f} MB")
    print(f"   - Validation data: {val_size:.1f} MB")
    print(f"\n🚀 Ready to start training!")
    print("Run: torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml")

if __name__ == "__main__":
    download_wikitext()