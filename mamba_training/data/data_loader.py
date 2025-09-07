"""Efficient data loading and batching system for Mamba training."""

import logging
import random
from typing import List, Dict, Any, Optional, Iterator, Tuple, Union
from dataclasses import dataclass
import math

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist

from .dataset_processor import ProcessedSample, ProcessedDataset
from ..config import DataConfig


logger = logging.getLogger(__name__)


@dataclass
class BatchedSample:
    """Container for a batched sample with padding and attention masks."""
    input_ids: torch.Tensor  # [batch_size, max_seq_len]
    attention_mask: torch.Tensor  # [batch_size, max_seq_len]
    labels: Optional[torch.Tensor] = None  # [batch_size, max_seq_len]
    sequence_lengths: Optional[torch.Tensor] = None  # [batch_size]


class SequencePacker:
    """Packs multiple sequences into fixed-length chunks for efficient training."""
    
    def __init__(self, max_seq_length: int, pad_token_id: int = 0):
        """Initialize sequence packer.
        
        Args:
            max_seq_length: Maximum sequence length for packing
            pad_token_id: Token ID to use for padding
        """
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
    
    def pack_sequences(self, samples: List[ProcessedSample]) -> List[ProcessedSample]:
        """Pack multiple sequences into fixed-length chunks.
        
        Args:
            samples: List of processed samples to pack
            
        Returns:
            List of packed samples with optimal sequence utilization
        """
        if not samples:
            return []
        
        packed_samples = []
        current_chunk = []
        current_length = 0
        
        # Sort samples by length for better packing efficiency
        sorted_samples = sorted(samples, key=lambda x: len(x.input_ids))
        
        for sample in sorted_samples:
            sample_length = len(sample.input_ids)
            
            # If adding this sample would exceed max length, finalize current chunk
            if current_length + sample_length > self.max_seq_length:
                if current_chunk:
                    packed_sample = self._create_packed_sample(current_chunk)
                    packed_samples.append(packed_sample)
                    current_chunk = []
                    current_length = 0
            
            # If single sample is too long, truncate it
            if sample_length > self.max_seq_length:
                truncated_sample = ProcessedSample(
                    input_ids=sample.input_ids[:self.max_seq_length],
                    attention_mask=sample.attention_mask[:self.max_seq_length],
                    labels=sample.labels[:self.max_seq_length] if sample.labels else None,
                    metadata=sample.metadata
                )
                packed_samples.append(truncated_sample)
            else:
                current_chunk.append(sample)
                current_length += sample_length
        
        # Handle remaining chunk
        if current_chunk:
            packed_sample = self._create_packed_sample(current_chunk)
            packed_samples.append(packed_sample)
        
        logger.info(f"Packed {len(samples)} samples into {len(packed_samples)} chunks "
                   f"(efficiency: {len(samples)/len(packed_samples):.2f}x)")
        
        return packed_samples
    
    def _create_packed_sample(self, samples: List[ProcessedSample]) -> ProcessedSample:
        """Create a single packed sample from multiple samples.
        
        Args:
            samples: List of samples to pack together
            
        Returns:
            Single packed sample
        """
        if len(samples) == 1:
            return samples[0]
        
        # Concatenate all sequences with separator tokens if needed
        packed_input_ids = []
        packed_attention_mask = []
        packed_labels = []
        
        for i, sample in enumerate(samples):
            packed_input_ids.extend(sample.input_ids)
            packed_attention_mask.extend(sample.attention_mask)
            
            if sample.labels:
                packed_labels.extend(sample.labels)
            
            # Add separator between samples (except for the last one)
            if i < len(samples) - 1:
                # Use EOS token as separator if available, otherwise pad token
                separator_id = getattr(samples[0], 'eos_token_id', self.pad_token_id)
                packed_input_ids.append(separator_id)
                packed_attention_mask.append(1)
                if sample.labels:
                    packed_labels.append(separator_id)
        
        # Truncate if too long, then pad if necessary
        if len(packed_input_ids) > self.max_seq_length:
            packed_input_ids = packed_input_ids[:self.max_seq_length]
            packed_attention_mask = packed_attention_mask[:self.max_seq_length]
            if packed_labels:
                packed_labels = packed_labels[:self.max_seq_length]
        elif len(packed_input_ids) < self.max_seq_length:
            padding_length = self.max_seq_length - len(packed_input_ids)
            packed_input_ids.extend([self.pad_token_id] * padding_length)
            packed_attention_mask.extend([0] * padding_length)
            if packed_labels:
                packed_labels.extend([-100] * padding_length)  # -100 is ignored in loss
        
        return ProcessedSample(
            input_ids=packed_input_ids,
            attention_mask=packed_attention_mask,
            labels=packed_labels if packed_labels else None,
            metadata={'packed_samples': len(samples)}
        )


class DynamicBatchSampler(Sampler):
    """Sampler that creates batches with similar sequence lengths for efficiency."""
    
    def __init__(self, 
                 dataset: Dataset,
                 batch_size: int,
                 max_tokens: Optional[int] = None,
                 shuffle: bool = True,
                 drop_last: bool = False):
        """Initialize dynamic batch sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Target batch size
            max_tokens: Maximum tokens per batch (alternative to batch_size)
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get sequence lengths for efficient batching
        self.sequence_lengths = self._get_sequence_lengths()
        
    def _get_sequence_lengths(self) -> List[int]:
        """Get sequence lengths for all samples in the dataset."""
        lengths = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            if isinstance(sample, dict) and 'input_ids' in sample:
                lengths.append(len(sample['input_ids']))
            elif hasattr(sample, 'input_ids'):
                lengths.append(len(sample.input_ids))
            else:
                lengths.append(1)  # Fallback
        return lengths
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of indices."""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            random.shuffle(indices)
        
        # Group indices by similar sequence lengths
        if self.max_tokens:
            batches = self._create_token_based_batches(indices)
        else:
            batches = self._create_length_based_batches(indices)
        
        for batch in batches:
            yield batch
    
    def _create_length_based_batches(self, indices: List[int]) -> List[List[int]]:
        """Create batches grouped by similar sequence lengths."""
        # Sort indices by sequence length
        sorted_indices = sorted(indices, key=lambda i: self.sequence_lengths[i])
        
        batches = []
        for i in range(0, len(sorted_indices), self.batch_size):
            batch = sorted_indices[i:i + self.batch_size]
            if not self.drop_last or len(batch) == self.batch_size:
                batches.append(batch)
        
        # Shuffle batches to avoid length-based ordering in training
        if self.shuffle:
            random.shuffle(batches)
        
        return batches
    
    def _create_token_based_batches(self, indices: List[int]) -> List[List[int]]:
        """Create batches based on maximum token count."""
        # Sort indices by sequence length
        sorted_indices = sorted(indices, key=lambda i: self.sequence_lengths[i])
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx in sorted_indices:
            seq_len = self.sequence_lengths[idx]
            
            # Check if adding this sample would exceed token limit
            if current_batch and (current_tokens + seq_len > self.max_tokens):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(idx)
            current_tokens += seq_len
            
            # Also respect batch size limit
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
        
        # Add remaining batch
        if current_batch and (not self.drop_last or len(current_batch) == self.batch_size):
            batches.append(current_batch)
        
        # Shuffle batches
        if self.shuffle:
            random.shuffle(batches)
        
        return batches
    
    def __len__(self) -> int:
        """Get number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return math.ceil(len(self.dataset) / self.batch_size)


class DistributedBatchSampler(Sampler):
    """Distributed sampler for multi-GPU training."""
    
    def __init__(self,
                 sampler: Sampler,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0):
        """Initialize distributed batch sampler.
        
        Args:
            sampler: Base sampler to distribute
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        
        # Get all batches from base sampler
        self.batches = list(sampler)
        self.num_samples = len(self.batches)
        
        # Calculate samples per replica
        self.num_samples_per_replica = math.ceil(self.num_samples / self.num_replicas)
        self.total_size = self.num_samples_per_replica * self.num_replicas
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches for current rank."""
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.batches), generator=g).tolist()
        else:
            indices = list(range(len(self.batches)))
        
        # Add extra samples to make it evenly divisible
        indices += indices[:self.total_size - len(indices)]
        assert len(indices) == self.total_size
        
        # Subsample for current rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples_per_replica
        
        # Yield batches for current rank
        for idx in indices:
            yield self.batches[idx]
    
    def __len__(self) -> int:
        """Get number of batches for current rank."""
        return self.num_samples_per_replica
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> BatchedSample:
    """Collate function for batching samples with dynamic padding.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        BatchedSample with padded sequences
    """
    # Extract tensors from batch
    input_ids = [sample['input_ids'] for sample in batch]
    attention_masks = [sample['attention_mask'] for sample in batch]
    labels = [sample['labels'] for sample in batch if sample['labels'] is not None]
    
    # Get sequence lengths before padding
    sequence_lengths = torch.tensor([len(seq) for seq in input_ids])
    
    # Pad sequences to the same length
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    labels_padded = None
    if labels:
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return BatchedSample(
        input_ids=input_ids_padded,
        attention_mask=attention_masks_padded,
        labels=labels_padded,
        sequence_lengths=sequence_lengths
    )


class MambaDataLoader:
    """High-level data loader for Mamba training with advanced features."""
    
    def __init__(self,
                 dataset: Union[ProcessedDataset, Dataset],
                 config: DataConfig,
                 batch_size: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 shuffle: bool = True,
                 num_workers: Optional[int] = None,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 use_sequence_packing: bool = False,
                 distributed: bool = False):
        """Initialize Mamba data loader.
        
        Args:
            dataset: Dataset to load from
            config: Data configuration
            batch_size: Batch size (uses config if None)
            max_tokens: Maximum tokens per batch
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            drop_last: Whether to drop last incomplete batch
            use_sequence_packing: Whether to use sequence packing
            distributed: Whether to use distributed sampling
        """
        self.dataset = dataset
        self.config = config
        self.batch_size = batch_size or config.preprocessing_batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.num_workers = num_workers or config.num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_sequence_packing = use_sequence_packing
        self.distributed = distributed
        
        # Apply sequence packing if requested
        if use_sequence_packing and hasattr(dataset, 'samples'):
            packer = SequencePacker(config.max_seq_length)
            packed_samples = packer.pack_sequences(dataset.samples)
            self.dataset = ProcessedDataset(packed_samples)
        
        # Create sampler
        self.sampler = self._create_sampler()
        
        # Create PyTorch DataLoader
        self.dataloader = TorchDataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
        
        logger.info(f"Created MambaDataLoader with {len(self.dataset)} samples, "
                   f"batch_size={self.batch_size}, num_workers={self.num_workers}")
    
    def _create_sampler(self) -> Sampler:
        """Create appropriate sampler based on configuration."""
        # Create base sampler
        base_sampler = DynamicBatchSampler(
            dataset=self.dataset,
            batch_size=self.batch_size,
            max_tokens=self.max_tokens,
            shuffle=self.shuffle,
            drop_last=self.drop_last
        )
        
        # Wrap with distributed sampler if needed
        if self.distributed:
            return DistributedBatchSampler(
                sampler=base_sampler,
                shuffle=self.shuffle
            )
        
        return base_sampler
    
    def __iter__(self) -> Iterator[BatchedSample]:
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.sampler)
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for distributed training."""
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get estimated memory usage statistics."""
        if not hasattr(self.dataset, '__getitem__'):
            return {}
        
        # Sample a few items to estimate memory usage
        sample_indices = random.sample(range(len(self.dataset)), min(10, len(self.dataset)))
        total_tokens = 0
        total_samples = 0
        
        for idx in sample_indices:
            sample = self.dataset[idx]
            if isinstance(sample, dict) and 'input_ids' in sample:
                total_tokens += len(sample['input_ids'])
            elif hasattr(sample, 'input_ids'):
                total_tokens += len(sample.input_ids)
            total_samples += 1
        
        avg_tokens_per_sample = total_tokens / total_samples if total_samples > 0 else 0
        avg_tokens_per_batch = avg_tokens_per_sample * self.batch_size
        
        # Estimate memory usage (rough approximation)
        # Each token uses ~4 bytes for input_ids + 4 bytes for attention_mask + 4 bytes for labels
        bytes_per_token = 12
        estimated_batch_memory_mb = (avg_tokens_per_batch * bytes_per_token) / (1024 * 1024)
        
        return {
            'avg_tokens_per_sample': avg_tokens_per_sample,
            'avg_tokens_per_batch': avg_tokens_per_batch,
            'estimated_batch_memory_mb': estimated_batch_memory_mb,
            'num_batches': len(self),
            'total_samples': len(self.dataset)
        }


def create_data_loaders(train_dataset: Dataset,
                       val_dataset: Optional[Dataset],
                       config: DataConfig,
                       distributed: bool = False) -> Tuple[MambaDataLoader, Optional[MambaDataLoader]]:
    """Create train and validation data loaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        config: Data configuration
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create training data loader
    train_loader = MambaDataLoader(
        dataset=train_dataset,
        config=config,
        shuffle=True,
        use_sequence_packing=True,
        distributed=distributed,
        drop_last=True
    )
    
    # Create validation data loader
    val_loader = None
    if val_dataset is not None:
        val_loader = MambaDataLoader(
            dataset=val_dataset,
            config=config,
            shuffle=False,
            use_sequence_packing=False,  # Don't pack validation data
            distributed=distributed,
            drop_last=False
        )
    
    return train_loader, val_loader