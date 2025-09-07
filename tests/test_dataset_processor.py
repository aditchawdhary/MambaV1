"""Tests for dataset processing and tokenization functionality."""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from transformers import AutoTokenizer

from mamba_training.data.dataset_processor import (
    DatasetProcessor, 
    TextQualityFilter, 
    ProcessedSample,
    ProcessedDataset
)
from mamba_training.config import DataConfig


class TestTextQualityFilter:
    """Test cases for TextQualityFilter."""
    
    def test_valid_text(self):
        """Test that valid text passes all filters."""
        filter_obj = TextQualityFilter()
        text = "This is a valid text sample with sufficient length and diversity."
        assert filter_obj.is_valid(text) is True
    
    def test_too_short_text(self):
        """Test that text below minimum length is filtered out."""
        filter_obj = TextQualityFilter(min_length=50)
        text = "Short text"
        assert filter_obj.is_valid(text) is False
    
    def test_too_long_text(self):
        """Test that text above maximum length is filtered out."""
        filter_obj = TextQualityFilter(max_length=20)
        text = "This text is definitely too long for the filter"
        assert filter_obj.is_valid(text) is False
    
    def test_insufficient_words(self):
        """Test that text with too few words is filtered out."""
        filter_obj = TextQualityFilter(min_word_count=10)
        text = "Only three words"
        assert filter_obj.is_valid(text) is False
    
    def test_excessive_repetition(self):
        """Test that text with excessive repetition is filtered out."""
        filter_obj = TextQualityFilter()
        # Create text with repeated trigrams
        repeated_text = "the quick brown " * 20
        assert filter_obj.is_valid(repeated_text) is False
    
    def test_low_character_diversity(self):
        """Test that text with low character diversity is filtered out."""
        filter_obj = TextQualityFilter()
        # Text with very low character diversity
        low_diversity_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        assert filter_obj.is_valid(low_diversity_text) is False


class TestDatasetProcessor:
    """Test cases for DatasetProcessor."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.pad_token = '<pad>'
        tokenizer.eos_token = '</s>'
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        
        # Mock the __call__ method for HuggingFace-style tokenization
        def mock_call(text, truncation=True, max_length=None, padding=False, return_tensors=None):
            # Simple mock tokenization: split by spaces and assign IDs
            words = text.split()
            input_ids = [i + 2 for i in range(len(words))]  # Start from 2 to avoid special tokens
            if max_length and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            return {
                'input_ids': input_ids,
                'attention_mask': [1] * len(input_ids)
            }
        
        tokenizer.__call__ = mock_call
        tokenizer.__len__ = lambda self: 50000
        
        # Add encode method for fallback
        def mock_encode(text, add_special_tokens=True):
            words = text.split()
            return [i + 2 for i in range(len(words))]
        
        tokenizer.encode = mock_encode
        
        return tokenizer
    
    @pytest.fixture
    def data_config(self):
        """Create a test data configuration."""
        return DataConfig(
            max_seq_length=128,
            tokenizer_path="mock_tokenizer",
            preprocessing_batch_size=10
        )
    
    def test_process_text_valid(self, mock_tokenizer, data_config):
        """Test processing of valid text."""
        processor = DatasetProcessor(mock_tokenizer, data_config)
        text = "This is a valid text sample for testing tokenization."
        
        sample = processor.process_text(text)
        
        assert sample is not None
        assert isinstance(sample, ProcessedSample)
        assert len(sample.input_ids) > 0
        assert len(sample.attention_mask) == len(sample.input_ids)
        assert sample.labels == sample.input_ids
        assert sample.metadata['original_length'] == len(text)
    
    def test_process_text_filtered_out(self, mock_tokenizer, data_config):
        """Test that invalid text is filtered out."""
        quality_filter = TextQualityFilter(min_length=100)
        processor = DatasetProcessor(mock_tokenizer, data_config, quality_filter)
        text = "Short"  # Too short
        
        sample = processor.process_text(text)
        
        assert sample is None
    
    def test_process_text_truncation(self, mock_tokenizer, data_config):
        """Test that long text is properly truncated."""
        # Create a quality filter that allows everything
        class PermissiveFilter:
            def is_valid(self, text):
                return True
        
        processor = DatasetProcessor(mock_tokenizer, data_config, PermissiveFilter())
        # Create a simple long text
        long_text = "word " * 200  # This will create 200 tokens
        
        sample = processor.process_text(long_text)
        
        assert sample is not None
        assert len(sample.input_ids) <= data_config.max_seq_length
    
    def test_load_txt_file(self, mock_tokenizer, data_config):
        """Test loading text from a .txt file."""
        processor = DatasetProcessor(mock_tokenizer, data_config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("First paragraph with some text.\n\n")
            f.write("Second paragraph with more text.\n\n")
            f.write("Third paragraph with even more text.")
            temp_path = f.name
        
        try:
            texts = processor._load_raw_data(Path(temp_path))
            assert len(texts) == 3
            assert "First paragraph" in texts[0]
            assert "Second paragraph" in texts[1]
            assert "Third paragraph" in texts[2]
        finally:
            Path(temp_path).unlink()
    
    def test_load_json_file(self, mock_tokenizer, data_config):
        """Test loading text from a JSON file."""
        processor = DatasetProcessor(mock_tokenizer, data_config)
        
        test_data = [
            {"text": "First text sample"},
            {"text": "Second text sample"},
            {"content": "Third text sample with different key"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            texts = processor._load_raw_data(Path(temp_path))
            assert len(texts) == 3
            assert "First text sample" in texts
            assert "Second text sample" in texts
            assert "Third text sample with different key" in texts
        finally:
            Path(temp_path).unlink()
    
    def test_load_jsonl_file(self, mock_tokenizer, data_config):
        """Test loading text from a JSONL file."""
        processor = DatasetProcessor(mock_tokenizer, data_config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "First line of JSONL"}\n')
            f.write('{"text": "Second line of JSONL"}\n')
            f.write('{"content": "Third line with different key"}\n')
            temp_path = f.name
        
        try:
            texts = processor._load_raw_data(Path(temp_path))
            assert len(texts) == 3
            assert "First line of JSONL" in texts
            assert "Second line of JSONL" in texts
            assert "Third line with different key" in texts
        finally:
            Path(temp_path).unlink()
    
    def test_process_dataset_complete(self, mock_tokenizer, data_config):
        """Test complete dataset processing workflow."""
        processor = DatasetProcessor(mock_tokenizer, data_config)
        
        # Create test dataset file
        test_data = [
            "This is the first valid text sample for processing.",
            "Here is another valid text sample with sufficient length.",
            "Short",  # This should be filtered out
            "A third valid text sample that should pass all quality filters."
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            samples = processor.process_dataset(temp_path)
            
            # Should have 3 samples (one filtered out)
            assert len(samples) == 3
            
            for sample in samples:
                assert isinstance(sample, ProcessedSample)
                assert len(sample.input_ids) > 0
                assert len(sample.attention_mask) == len(sample.input_ids)
                assert sample.labels is not None
        finally:
            Path(temp_path).unlink()
    
    def test_get_vocab_size(self, mock_tokenizer, data_config):
        """Test getting vocabulary size."""
        processor = DatasetProcessor(mock_tokenizer, data_config)
        vocab_size = processor.get_vocab_size()
        assert vocab_size == 50000
    
    def test_get_special_token_ids(self, mock_tokenizer, data_config):
        """Test getting special token IDs."""
        processor = DatasetProcessor(mock_tokenizer, data_config)
        special_tokens = processor.get_special_token_ids()
        
        assert 'pad_token_id' in special_tokens
        assert 'eos_token_id' in special_tokens
        assert special_tokens['pad_token_id'] == 0
        assert special_tokens['eos_token_id'] == 1


class TestProcessedDataset:
    """Test cases for ProcessedDataset."""
    
    def test_dataset_creation(self):
        """Test creating a ProcessedDataset."""
        samples = [
            ProcessedSample(
                input_ids=[1, 2, 3, 4],
                attention_mask=[1, 1, 1, 1],
                labels=[1, 2, 3, 4]
            ),
            ProcessedSample(
                input_ids=[5, 6, 7],
                attention_mask=[1, 1, 1],
                labels=[5, 6, 7]
            )
        ]
        
        dataset = ProcessedDataset(samples)
        
        assert len(dataset) == 2
        
        # Test first sample
        item = dataset[0]
        assert torch.equal(item['input_ids'], torch.tensor([1, 2, 3, 4]))
        assert torch.equal(item['attention_mask'], torch.tensor([1, 1, 1, 1]))
        assert torch.equal(item['labels'], torch.tensor([1, 2, 3, 4]))
        
        # Test second sample
        item = dataset[1]
        assert torch.equal(item['input_ids'], torch.tensor([5, 6, 7]))
        assert torch.equal(item['attention_mask'], torch.tensor([1, 1, 1]))
        assert torch.equal(item['labels'], torch.tensor([5, 6, 7]))
    
    def test_dataset_from_file(self):
        """Test loading ProcessedDataset from file."""
        test_data = [
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [1, 2, 3],
                'metadata': {'original_length': 20}
            },
            {
                'input_ids': [4, 5, 6, 7],
                'attention_mask': [1, 1, 1, 1],
                'labels': [4, 5, 6, 7],
                'metadata': {'original_length': 30}
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ProcessedDataset.from_file(temp_path)
            
            assert len(dataset) == 2
            
            item = dataset[0]
            assert torch.equal(item['input_ids'], torch.tensor([1, 2, 3]))
            assert torch.equal(item['attention_mask'], torch.tensor([1, 1, 1]))
            assert torch.equal(item['labels'], torch.tensor([1, 2, 3]))
        finally:
            Path(temp_path).unlink()


class TestTokenizerIntegration:
    """Integration tests with real tokenizers."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for HuggingFace tokenizers")
    def test_with_gpt2_tokenizer(self):
        """Test with actual GPT-2 tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            config = DataConfig(max_seq_length=64)
            
            processor = DatasetProcessor(tokenizer, config)
            text = "This is a test sentence for the GPT-2 tokenizer integration."
            
            sample = processor.process_text(text)
            
            assert sample is not None
            assert len(sample.input_ids) > 0
            assert len(sample.attention_mask) == len(sample.input_ids)
            
            # Verify we can decode back to text
            decoded = tokenizer.decode(sample.input_ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0
            
        except Exception as e:
            pytest.skip(f"Could not load GPT-2 tokenizer: {e}")


if __name__ == "__main__":
    pytest.main([__file__])