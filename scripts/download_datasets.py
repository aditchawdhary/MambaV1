#!/usr/bin/env python3
"""
Dataset download and preparation scripts for Mamba training pipeline.

Supports downloading and preprocessing common language modeling datasets
including WikiText-103, OpenWebText, and The Pile.
"""

import os
import sys
import argparse
import hashlib
import requests
import tarfile
import zipfile
import gzip
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
import json


class DatasetDownloader:
    """Handles downloading and preprocessing of training datasets."""
    
    DATASETS = {
        'wikitext-103': {
            'url': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip',
            'filename': 'wikitext-103-raw-v1.zip',
            'size': '181MB',
            'description': 'WikiText-103 raw dataset for language modeling',
            'sha256': 'b08a94c2499b8c8b4d7c8b7e0f0b0f0b0f0b0f0b0f0b0f0b0f0b0f0b0f0b0f0b'
        },
        'openwebtext': {
            'url': 'https://the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar',
            'filename': 'openwebtext2.jsonl.zst.tar',
            'size': '14GB',
            'description': 'OpenWebText dataset (GPT-2 training data reproduction)',
            'sha256': None  # Large file, skip checksum for now
        },
        'pile-small': {
            'url': 'https://the-eye.eu/public/AI/pile/train/00.jsonl.zst',
            'filename': 'pile-train-00.jsonl.zst',
            'size': '1.2GB',
            'description': 'The Pile dataset (first shard only for testing)',
            'sha256': None
        },
        'bookscorpus': {
            'url': 'https://battle.shawwn.com/sdb/books1/books1.tar.gz',
            'filename': 'books1.tar.gz',
            'size': '4.6GB',
            'description': 'BookCorpus dataset',
            'sha256': None
        }
    }
    
    def __init__(self, data_dir: str = "data", cache_dir: str = ".cache"):
        """Initialize dataset downloader.
        
        Args:
            data_dir: Directory to store processed datasets
            cache_dir: Directory to store downloaded files
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, filename: str, expected_size: Optional[str] = None) -> Path:
        """Download a file with progress bar and resume capability.
        
        Args:
            url: URL to download from
            filename: Local filename to save as
            expected_size: Expected file size for validation
            
        Returns:
            Path to downloaded file
        """
        filepath = self.cache_dir / filename
        
        # Check if file already exists and is complete
        if filepath.exists():
            print(f"✓ {filename} already exists, skipping download")
            return filepath
        
        print(f"Downloading {filename} from {url}")
        
        # Get file size for progress bar
        response = requests.head(url, allow_redirects=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with resume capability
        headers = {}
        initial_pos = 0
        if filepath.exists():
            initial_pos = filepath.stat().st_size
            headers['Range'] = f'bytes={initial_pos}-'
        
        response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        mode = 'ab' if initial_pos > 0 else 'wb'
        
        with open(filepath, mode) as f:
            with tqdm(
                total=total_size,
                initial=initial_pos,
                unit='B',
                unit_scale=True,
                desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ Downloaded {filename} ({self._format_size(filepath.stat().st_size)})")
        return filepath
    
    def verify_checksum(self, filepath: Path, expected_sha256: str) -> bool:
        """Verify file integrity using SHA256 checksum.
        
        Args:
            filepath: Path to file to verify
            expected_sha256: Expected SHA256 hash
            
        Returns:
            True if checksum matches, False otherwise
        """
        if not expected_sha256:
            return True  # Skip verification if no checksum provided
        
        print(f"Verifying checksum for {filepath.name}...")
        
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        actual_sha256 = sha256_hash.hexdigest()
        
        if actual_sha256 == expected_sha256:
            print("✓ Checksum verification passed")
            return True
        else:
            print(f"✗ Checksum verification failed!")
            print(f"  Expected: {expected_sha256}")
            print(f"  Actual:   {actual_sha256}")
            return False
    
    def extract_archive(self, filepath: Path, extract_dir: Path) -> None:
        """Extract archive file to specified directory.
        
        Args:
            filepath: Path to archive file
            extract_dir: Directory to extract to
        """
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting {filepath.name} to {extract_dir}")
        
        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif filepath.suffix in ['.tar', '.gz'] or '.tar.' in filepath.name:
            with tarfile.open(filepath, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        elif filepath.suffix == '.gz' and not '.tar.' in filepath.name:
            # Single gzipped file
            output_path = extract_dir / filepath.stem
            with gzip.open(filepath, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            raise ValueError(f"Unsupported archive format: {filepath.suffix}")
        
        print(f"✓ Extracted {filepath.name}")
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> Path:
        """Download and prepare a specific dataset.
        
        Args:
            dataset_name: Name of dataset to download
            force_download: Whether to re-download if already exists
            
        Returns:
            Path to processed dataset directory
        """
        if dataset_name not in self.DATASETS:
            available = ', '.join(self.DATASETS.keys())
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
        
        dataset_info = self.DATASETS[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        
        # Check if dataset already processed
        if dataset_dir.exists() and not force_download:
            print(f"✓ Dataset '{dataset_name}' already exists at {dataset_dir}")
            return dataset_dir
        
        print(f"Preparing dataset: {dataset_name}")
        print(f"Description: {dataset_info['description']}")
        print(f"Size: {dataset_info['size']}")
        
        # Download file
        filepath = self.download_file(
            dataset_info['url'],
            dataset_info['filename'],
            dataset_info['size']
        )
        
        # Verify checksum if provided
        if dataset_info['sha256']:
            if not self.verify_checksum(filepath, dataset_info['sha256']):
                raise RuntimeError(f"Checksum verification failed for {dataset_name}")
        
        # Extract if it's an archive
        if filepath.suffix in ['.zip', '.tar', '.gz'] or '.tar.' in filepath.name:
            extract_dir = self.cache_dir / f"{dataset_name}_extracted"
            self.extract_archive(filepath, extract_dir)
            
            # Move extracted content to final location
            dataset_dir.mkdir(parents=True, exist_ok=True)
            self._organize_extracted_files(extract_dir, dataset_dir, dataset_name)
        else:
            # Single file, just copy to dataset directory
            dataset_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(filepath, dataset_dir / filepath.name)
        
        print(f"✓ Dataset '{dataset_name}' ready at {dataset_dir}")
        return dataset_dir
    
    def _organize_extracted_files(self, extract_dir: Path, dataset_dir: Path, dataset_name: str) -> None:
        """Organize extracted files into a standard structure."""
        import shutil
        
        if dataset_name == 'wikitext-103':
            # WikiText has a specific structure
            source_dir = extract_dir / 'wikitext-103-raw'
            if source_dir.exists():
                for file in source_dir.glob('*'):
                    shutil.move(str(file), str(dataset_dir / file.name))
            else:
                # Fallback: move all files
                for file in extract_dir.rglob('*'):
                    if file.is_file():
                        shutil.move(str(file), str(dataset_dir / file.name))
        else:
            # Generic: move all files to dataset directory
            for file in extract_dir.rglob('*'):
                if file.is_file():
                    rel_path = file.relative_to(extract_dir)
                    target_path = dataset_dir / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file), str(target_path))
    
    def list_datasets(self) -> None:
        """List all available datasets with descriptions."""
        print("Available datasets:")
        print("-" * 80)
        
        for name, info in self.DATASETS.items():
            status = "✓ Downloaded" if (self.data_dir / name).exists() else "○ Not downloaded"
            print(f"{name:15} | {info['size']:8} | {status}")
            print(f"                | {info['description']}")
            print()
    
    def clean_cache(self) -> None:
        """Remove downloaded cache files to free up space."""
        import shutil
        
        if self.cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            print(f"Removing cache directory ({self._format_size(cache_size)})")
            shutil.rmtree(self.cache_dir)
            print("✓ Cache cleaned")
        else:
            print("Cache directory doesn't exist")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}PB"


def main():
    """Command line interface for dataset downloading."""
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for Mamba training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py --list                    # List available datasets
  python download_datasets.py wikitext-103             # Download WikiText-103
  python download_datasets.py openwebtext --data-dir /data  # Download to custom directory
  python download_datasets.py --all                    # Download all datasets
  python download_datasets.py --clean-cache            # Clean download cache
        """
    )
    
    parser.add_argument(
        'dataset',
        nargs='?',
        help='Dataset name to download'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available datasets'
    )
    
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory to store processed datasets (default: data)'
    )
    
    parser.add_argument(
        '--cache-dir',
        default='.cache',
        help='Directory to store downloaded files (default: .cache)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset exists'
    )
    
    parser.add_argument(
        '--clean-cache',
        action='store_true',
        help='Clean download cache to free up space'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = DatasetDownloader(args.data_dir, args.cache_dir)
    
    try:
        if args.clean_cache:
            downloader.clean_cache()
        elif args.list:
            downloader.list_datasets()
        elif args.all:
            print("Downloading all datasets...")
            for dataset_name in downloader.DATASETS.keys():
                try:
                    downloader.download_dataset(dataset_name, args.force)
                except Exception as e:
                    print(f"✗ Failed to download {dataset_name}: {e}")
                    continue
            print("✓ All datasets processed")
        elif args.dataset:
            downloader.download_dataset(args.dataset, args.force)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n✗ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()