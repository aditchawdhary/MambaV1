# Dataset Management Scripts

This directory contains scripts for downloading, preprocessing, and managing training datasets for the Mamba training pipeline.

## Quick Start

### Download a Dataset

```bash
# List available datasets
python scripts/download_datasets.py --list

# Download WikiText-103 (good for testing)
python scripts/download_datasets.py wikitext-103

# Download OpenWebText (larger, production-ready)
python scripts/download_datasets.py openwebtext

# Download all datasets
python scripts/download_datasets.py --all
```

### Custom Data Directory

```bash
# Download to custom directory
python scripts/download_datasets.py wikitext-103 --data-dir /path/to/data

# Use custom cache directory
python scripts/download_datasets.py openwebtext --cache-dir /tmp/downloads
```

## Available Datasets

| Dataset | Size | Description | Best For |
|---------|------|-------------|----------|
| **wikitext-103** | 181MB | Clean Wikipedia text | Development, testing |
| **openwebtext** | 14GB | GPT-2 training data reproduction | Production training |
| **pile-small** | 1.2GB | First shard of The Pile | Medium-scale experiments |
| **bookscorpus** | 4.6GB | Collection of 11,000+ books | Literary text training |

### Dataset Details

#### WikiText-103
- **Size**: 181MB compressed
- **Content**: Clean, well-formatted Wikipedia articles
- **Use Case**: Perfect for development and testing
- **Pros**: Small, clean, fast to download
- **Cons**: Limited diversity

#### OpenWebText
- **Size**: 14GB compressed
- **Content**: Web pages used to train GPT-2
- **Use Case**: Production training, reproducing GPT-2 results
- **Pros**: High quality, diverse content
- **Cons**: Large download, requires preprocessing

#### The Pile (Small)
- **Size**: 1.2GB (first shard only)
- **Content**: Diverse text from books, papers, code, etc.
- **Use Case**: Medium-scale experiments
- **Pros**: Very diverse content types
- **Cons**: Single shard limits scale

#### BookCorpus
- **Size**: 4.6GB
- **Content**: Over 11,000 books from various genres
- **Use Case**: Training on literary text
- **Pros**: Coherent long-form text
- **Cons**: Less diverse than web text

## Usage Examples

### Basic Usage

```bash
# Download and prepare WikiText-103
python scripts/download_datasets.py wikitext-103

# Check what's available
python scripts/download_datasets.py --list

# Force re-download even if exists
python scripts/download_datasets.py wikitext-103 --force
```

### Advanced Usage

```bash
# Download multiple datasets for comparison
python scripts/download_datasets.py wikitext-103
python scripts/download_datasets.py pile-small

# Clean up download cache to save space
python scripts/download_datasets.py --clean-cache

# Download to network storage
python scripts/download_datasets.py openwebtext --data-dir /mnt/shared/datasets
```

### Integration with Training

```bash
# Download dataset and start training
python scripts/download_datasets.py wikitext-103 --data-dir data/
python train.py --dataset-path data/wikitext-103/
```

## Dataset Structure

After downloading, datasets are organized as follows:

```
data/
├── wikitext-103/
│   ├── wiki.train.raw
│   ├── wiki.valid.raw
│   └── wiki.test.raw
├── openwebtext/
│   ├── openwebtext2.jsonl
│   └── metadata.json
├── pile-small/
│   └── pile-train-00.jsonl
└── bookscorpus/
    └── books1/
        ├── book1.txt
        ├── book2.txt
        └── ...
```

## Features

### Resume Downloads
- Automatically resumes interrupted downloads
- Checks existing files to avoid re-downloading
- Progress bars show download status

### Integrity Verification
- SHA256 checksum verification (when available)
- File size validation
- Corruption detection

### Efficient Storage
- Automatic extraction of compressed files
- Organized directory structure
- Cache management to save space

### Error Handling
- Retry logic for network failures
- Graceful handling of corrupted downloads
- Clear error messages and recovery suggestions

## Configuration

### Environment Variables

```bash
# Set default data directory
export MAMBA_DATA_DIR="/path/to/datasets"

# Set cache directory
export MAMBA_CACHE_DIR="/tmp/mamba_cache"

# Set download timeout
export MAMBA_DOWNLOAD_TIMEOUT=300
```

### Custom Dataset Sources

You can extend the downloader to support custom datasets:

```python
# Add to DATASETS dictionary in download_datasets.py
'my-custom-dataset': {
    'url': 'https://example.com/dataset.tar.gz',
    'filename': 'dataset.tar.gz',
    'size': '2GB',
    'description': 'My custom training dataset',
    'sha256': 'abc123...'  # Optional checksum
}
```

## Performance Tips

### Faster Downloads
```bash
# Use multiple connections (if supported)
export MAMBA_PARALLEL_DOWNLOADS=4

# Use faster mirror (if available)
export MAMBA_MIRROR_URL="https://fast-mirror.example.com"
```

### Storage Optimization
```bash
# Clean cache after each download
python scripts/download_datasets.py wikitext-103
python scripts/download_datasets.py --clean-cache

# Use temporary cache directory
python scripts/download_datasets.py openwebtext --cache-dir /tmp/cache
```

### Network-Constrained Environments
```bash
# Download smaller dataset first
python scripts/download_datasets.py wikitext-103

# Download during off-peak hours
echo "0 2 * * * python /path/to/download_datasets.py openwebtext" | crontab -
```

## Preprocessing Integration

The download script integrates with the training pipeline's preprocessing:

```python
from mamba_training.data import DatasetProcessor
from scripts.download_datasets import DatasetDownloader

# Download and preprocess in one step
downloader = DatasetDownloader()
dataset_path = downloader.download_dataset('wikitext-103')

processor = DatasetProcessor(tokenizer_path='tokenizer.model')
processed_dataset = processor.process_dataset(dataset_path)
```

## Troubleshooting

### Common Issues

**1. Download Fails**
```bash
# Check network connection
curl -I https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip

# Try with different cache directory
python scripts/download_datasets.py wikitext-103 --cache-dir /tmp/new_cache
```

**2. Checksum Verification Fails**
```bash
# Force re-download
python scripts/download_datasets.py wikitext-103 --force

# Skip checksum verification (not recommended)
# Edit download_datasets.py and set sha256 to None
```

**3. Extraction Fails**
```bash
# Check available disk space
df -h

# Try manual extraction
cd .cache
tar -tf wikitext-103-raw-v1.zip  # Check archive integrity
```

**4. Permission Errors**
```bash
# Fix permissions
sudo chown -R $USER:$USER data/
chmod -R 755 data/

# Use different directory
python scripts/download_datasets.py wikitext-103 --data-dir ~/datasets
```

### Debug Mode

Enable verbose logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run download script
python scripts/download_datasets.py wikitext-103
```

## Integration with Cloud Storage

### Upload to S3

```bash
# Download locally then upload to S3
python scripts/download_datasets.py wikitext-103
aws s3 sync data/wikitext-103/ s3://your-bucket/datasets/wikitext-103/

# Download directly to S3-mounted directory
python scripts/download_datasets.py openwebtext --data-dir /mnt/s3/datasets/
```

### Download from S3

```bash
# Download preprocessed datasets from S3
aws s3 sync s3://your-bucket/datasets/wikitext-103/ data/wikitext-103/
```

## Custom Dataset Integration

### Adding New Datasets

1. **Add dataset configuration**:
```python
# In download_datasets.py, add to DATASETS dictionary
'my-dataset': {
    'url': 'https://example.com/my-dataset.tar.gz',
    'filename': 'my-dataset.tar.gz',
    'size': '1GB',
    'description': 'My custom dataset',
    'sha256': None  # Add checksum if available
}
```

2. **Add custom preprocessing** (if needed):
```python
def _organize_extracted_files(self, extract_dir: Path, dataset_dir: Path, dataset_name: str):
    if dataset_name == 'my-dataset':
        # Custom organization logic
        pass
    else:
        # Default logic
        pass
```

3. **Test the integration**:
```bash
python scripts/download_datasets.py my-dataset
```

### Dataset Validation

Create validation scripts for custom datasets:

```python
def validate_dataset(dataset_path):
    """Validate dataset format and content."""
    # Check file formats
    # Validate content structure
    # Verify data quality
    pass
```

## Automation

### Scheduled Downloads

```bash
# Download new datasets nightly
echo "0 2 * * * cd /path/to/project && python scripts/download_datasets.py --all" | crontab -
```

### CI/CD Integration

```yaml
# .github/workflows/prepare-data.yml
name: Prepare Training Data
on:
  workflow_dispatch:
    inputs:
      dataset:
        description: 'Dataset to download'
        required: true
        default: 'wikitext-103'

jobs:
  download:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Download dataset
        run: python scripts/download_datasets.py ${{ github.event.inputs.dataset }}
      - name: Upload to S3
        run: aws s3 sync data/ s3://your-bucket/datasets/
```

## Best Practices

1. **Start Small**: Begin with WikiText-103 for development
2. **Verify Checksums**: Always verify data integrity when possible
3. **Clean Cache**: Regularly clean download cache to save space
4. **Monitor Usage**: Track download bandwidth and storage costs
5. **Backup Data**: Keep copies of processed datasets
6. **Version Control**: Track dataset versions and preprocessing changes

## Support

For dataset-related issues:
1. Check the troubleshooting section above
2. Verify network connectivity and permissions
3. Check available disk space
4. Review dataset source documentation
5. Consider using smaller datasets for testing