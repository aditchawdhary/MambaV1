"""Dataset processing and tokenization utilities for Mamba training."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
from dataclasses import dataclass
import re

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import sentencepiece as spm

from ..config import DataConfig


logger = logging.getLogger(__name__)


@dataclass
class ProcessedSample:
    """Container for a processed text sample."""
    input_ids: List[int]
    attention_mask: List[int]
    labels: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None


class TextQualityFilter:
    """Filters for text quality and content validation."""
    
    def __init__(self, 
                 min_length: int = 10,
                 max_length: int = 100000,
                 min_word_count: int = 5,
                 max_repetition_ratio: float = 0.8):
        """Initialize text quality filters.
        
        Args:
            min_length: Minimum character length
            max_length: Maximum character length
            min_word_count: Minimum word count
            max_repetition_ratio: Maximum ratio of repeated n-grams
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_word_count = min_word_count
        self.max_repetition_ratio = max_repetition_ratio
    
    def is_valid(self, text: str) -> bool:
        """Check if text passes quality filters.
        
        Args:
            text: Text to validate
            
        Returns:
            bool: True if text passes all filters
        """
        # Length checks
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # Word count check
        words = text.split()
        if len(words) < self.min_word_count:
            return False
        
        # Repetition check
        if self._has_excessive_repetition(text):
            return False
        
        # Character diversity check
        if self._has_low_character_diversity(text):
            return False
        
        return True
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive repetition in text."""
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        # Check for repeated 3-grams
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        if len(trigrams) == 0:
            return False
        
        unique_trigrams = set(trigrams)
        repetition_ratio = 1 - (len(unique_trigrams) / len(trigrams))
        
        return repetition_ratio > self.max_repetition_ratio
    
    def _has_low_character_diversity(self, text: str) -> bool:
        """Check for low character diversity."""
        if len(text) < 50:
            return False
        
        unique_chars = len(set(text.lower()))
        diversity_ratio = unique_chars / len(text)
        
        # Require at least 2% character diversity
        return diversity_ratio < 0.02


class DatasetProcessor:
    """Processes raw text datasets for Mamba model training."""
    
    def __init__(self, 
                 tokenizer: Union[str, PreTrainedTokenizer],
                 config: DataConfig,
                 quality_filter: Optional[TextQualityFilter] = None):
        """Initialize dataset processor.
        
        Args:
            tokenizer: Tokenizer instance or path to tokenizer
            config: Data configuration
            quality_filter: Optional text quality filter
        """
        self.config = config
        self.tokenizer = self._load_tokenizer(tokenizer)
        self.quality_filter = quality_filter or TextQualityFilter()
        
        # Ensure tokenizer has required special tokens
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized DatasetProcessor with tokenizer vocab size: {len(self.tokenizer)}")
    
    def _load_tokenizer(self, tokenizer: Union[str, PreTrainedTokenizer]) -> PreTrainedTokenizer:
        """Load tokenizer from path or return existing instance.
        
        Args:
            tokenizer: Tokenizer instance or path
            
        Returns:
            PreTrainedTokenizer: Loaded tokenizer
        """
        if isinstance(tokenizer, str):
            tokenizer_path = Path(tokenizer)
            
            # Try loading as HuggingFace tokenizer first
            try:
                return AutoTokenizer.from_pretrained(tokenizer)
            except Exception:
                pass
            
            # Try loading as SentencePiece model
            if tokenizer_path.exists() and tokenizer_path.suffix == '.model':
                try:
                    sp_model = spm.SentencePieceProcessor()
                    sp_model.load(str(tokenizer_path))
                    # Wrap SentencePiece in a compatible interface
                    return self._wrap_sentencepiece(sp_model)
                except Exception as e:
                    logger.error(f"Failed to load SentencePiece tokenizer: {e}")
            
            raise ValueError(f"Could not load tokenizer from: {tokenizer}")
        
        return tokenizer
    
    def _wrap_sentencepiece(self, sp_model: spm.SentencePieceProcessor) -> PreTrainedTokenizer:
        """Wrap SentencePiece model in HuggingFace-compatible interface."""
        # For now, we'll use a simple wrapper
        # In practice, you might want to use a more sophisticated wrapper
        class SPTokenizerWrapper:
            def __init__(self, sp_model):
                self.sp_model = sp_model
                self.pad_token = '<pad>'
                self.eos_token = '</s>'
                self.bos_token = '<s>'
                self.unk_token = '<unk>'
                self.pad_token_id = sp_model.pad_id()
                self.eos_token_id = sp_model.eos_id()
                self.bos_token_id = sp_model.bos_id()
                self.unk_token_id = sp_model.unk_id()
            
            def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
                return self.sp_model.encode(text, add_bos=add_special_tokens, add_eos=add_special_tokens)
            
            def decode(self, token_ids: List[int]) -> str:
                return self.sp_model.decode(token_ids)
            
            def __len__(self):
                return self.sp_model.vocab_size()
        
        return SPTokenizerWrapper(sp_model)
    
    def process_text(self, text: str) -> Optional[ProcessedSample]:
        """Process a single text sample.
        
        Args:
            text: Raw text to process
            
        Returns:
            ProcessedSample: Processed sample or None if filtered out
        """
        # Apply quality filters
        if not self.quality_filter.is_valid(text):
            return None
        
        # Tokenize text
        try:
            # Try HuggingFace-style tokenization first
            if hasattr(self.tokenizer, '__call__'):
                try:
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.config.max_seq_length,
                        padding=False,
                        return_tensors=None
                    )
                    input_ids = encoding['input_ids']
                    attention_mask = encoding.get('attention_mask', [1] * len(input_ids))
                except (TypeError, KeyError):
                    # Fallback to encode method
                    input_ids = self.tokenizer.encode(text)
                    # Truncate if necessary
                    if len(input_ids) > self.config.max_seq_length:
                        input_ids = input_ids[:self.config.max_seq_length]
                    attention_mask = [1] * len(input_ids)
            else:
                # Custom tokenizer
                input_ids = self.tokenizer.encode(text)
                # Truncate if necessary
                if len(input_ids) > self.config.max_seq_length:
                    input_ids = input_ids[:self.config.max_seq_length]
                attention_mask = [1] * len(input_ids)
            
            # For language modeling, labels are the same as input_ids
            labels = input_ids.copy()
            
            return ProcessedSample(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                metadata={'original_length': len(text)}
            )
            
        except Exception as e:
            logger.warning(f"Failed to tokenize text: {e}")
            return None
    
    def process_dataset(self, 
                       dataset_path: Union[str, Path],
                       output_path: Optional[Union[str, Path]] = None) -> List[ProcessedSample]:
        """Process a complete dataset from file.
        
        Args:
            dataset_path: Path to input dataset
            output_path: Optional path to save processed dataset
            
        Returns:
            List[ProcessedSample]: List of processed samples
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        logger.info(f"Processing dataset: {dataset_path}")
        
        # Load raw data based on file format
        raw_texts = self._load_raw_data(dataset_path)
        
        # Process texts in batches
        processed_samples = []
        batch_size = self.config.preprocessing_batch_size
        
        for i in range(0, len(raw_texts), batch_size):
            batch_texts = raw_texts[i:i + batch_size]
            batch_samples = []
            
            for text in batch_texts:
                sample = self.process_text(text)
                if sample is not None:
                    batch_samples.append(sample)
            
            processed_samples.extend(batch_samples)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_texts)}/{len(raw_texts)} samples, "
                           f"kept {len(processed_samples)} after filtering")
        
        logger.info(f"Dataset processing complete. Kept {len(processed_samples)}/{len(raw_texts)} samples "
                   f"({len(processed_samples)/len(raw_texts)*100:.1f}%)")
        
        # Save processed dataset if output path provided
        if output_path:
            self._save_processed_dataset(processed_samples, output_path)
        
        return processed_samples
    
    def _load_raw_data(self, dataset_path: Path) -> List[str]:
        """Load raw text data from various file formats.
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            List[str]: List of raw text samples
        """
        suffix = dataset_path.suffix.lower()
        
        if suffix == '.txt':
            return self._load_txt_file(dataset_path)
        elif suffix == '.json':
            return self._load_json_file(dataset_path)
        elif suffix == '.jsonl':
            return self._load_jsonl_file(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_txt_file(self, file_path: Path) -> List[str]:
        """Load text from a plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines to separate documents/paragraphs
        texts = [text.strip() for text in content.split('\n\n') if text.strip()]
        return texts
    
    def _load_json_file(self, file_path: Path) -> List[str]:
        """Load text from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of strings or objects
            texts = []
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    # Look for common text fields
                    text = item.get('text') or item.get('content') or item.get('body')
                    if text:
                        texts.append(text)
            return texts
        elif isinstance(data, dict):
            # Single object or nested structure
            if 'texts' in data:
                return data['texts']
            elif 'data' in data:
                return self._extract_texts_from_list(data['data'])
            else:
                # Try to find text in the object
                text = data.get('text') or data.get('content') or data.get('body')
                return [text] if text else []
        
        return []
    
    def _load_jsonl_file(self, file_path: Path) -> List[str]:
        """Load text from a JSONL file."""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    text = item.get('text') or item.get('content') or item.get('body')
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue
        return texts
    
    def _extract_texts_from_list(self, data_list: List[Any]) -> List[str]:
        """Extract text from a list of mixed data types."""
        texts = []
        for item in data_list:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                text = item.get('text') or item.get('content') or item.get('body')
                if text:
                    texts.append(text)
        return texts
    
    def _save_processed_dataset(self, 
                               samples: List[ProcessedSample], 
                               output_path: Union[str, Path]) -> None:
        """Save processed dataset to file.
        
        Args:
            samples: Processed samples to save
            output_path: Path to save the dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert samples to serializable format
        serializable_data = []
        for sample in samples:
            serializable_data.append({
                'input_ids': sample.input_ids,
                'attention_mask': sample.attention_mask,
                'labels': sample.labels,
                'metadata': sample.metadata
            })
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Saved {len(samples)} processed samples to {output_path}")
    
    def get_vocab_size(self) -> int:
        """Get tokenizer vocabulary size."""
        return len(self.tokenizer)
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs from tokenizer."""
        special_tokens = {}
        
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            special_tokens['pad_token_id'] = self.tokenizer.pad_token_id
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            special_tokens['eos_token_id'] = self.tokenizer.eos_token_id
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            special_tokens['bos_token_id'] = self.tokenizer.bos_token_id
        if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
            special_tokens['unk_token_id'] = self.tokenizer.unk_token_id
        
        return special_tokens


class ProcessedDataset(Dataset):
    """PyTorch Dataset for processed text samples."""
    
    def __init__(self, samples: List[ProcessedSample]):
        """Initialize dataset with processed samples.
        
        Args:
            samples: List of processed samples
        """
        self.samples = samples
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing tensors for input_ids, attention_mask, and labels
        """
        sample = self.samples[idx]
        
        return {
            'input_ids': torch.tensor(sample.input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(sample.attention_mask, dtype=torch.long),
            'labels': torch.tensor(sample.labels, dtype=torch.long) if sample.labels else None
        }
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'ProcessedDataset':
        """Load processed dataset from file.
        
        Args:
            file_path: Path to processed dataset file
            
        Returns:
            ProcessedDataset: Loaded dataset
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = ProcessedSample(
                input_ids=item['input_ids'],
                attention_mask=item['attention_mask'],
                labels=item.get('labels'),
                metadata=item.get('metadata')
            )
            samples.append(sample)
        
        return cls(samples)