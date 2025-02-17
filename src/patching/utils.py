import torch
import numpy as np
from typing import List, Tuple
from transformer_lens import HookedTransformer
from transformer_lens.components import Embed
from transformers import AutoTokenizer
import einops


def strip_eot(tensor, pad_token):
    mask = tensor != pad_token 
    return tensor[mask]  # Directly return filtered tensor

def align_batch_tensors(clean_tensors: torch.Tensor, corrupted_tensors: torch.Tensor, padding_token: int) -> (torch.Tensor, torch.Tensor):
    """
    Aligns two batched tensors by removing excess padding tokens (EOT tokens) from the left of the longer tensor for each sequence.

    Args:
        clean_tensors (torch.Tensor): The batch of shorter tensors. Shape: (batch_size, seq_len1)
        corrupted_tensors (torch.Tensor): The batch of longer tensors. Shape: (batch_size, seq_len2)
        padding_token (int): The padding token to remove. Default is 50256 (EOT token).

    Returns:
        torch.Tensor, torch.Tensor: Aligned tensors with the same shape for each batch.
    """
    aligned_clean = []
    aligned_corrupted = []
    
    for clean, corrupted in zip(clean_tensors, corrupted_tensors):
        # Calculate the difference in lengths for each sequence
        length_difference = corrupted.shape[0] - clean.shape[0]
        
        # Slice the extra padding tokens from the left of the corrupted tensor
        if length_difference > 0:
            corrupted = corrupted[length_difference:]
        
        # Ensure they are now the same length
        assert clean.shape[0] == corrupted.shape[0], "Sequences are not aligned correctly!"
        
        # Append the aligned sequences
        aligned_clean.append(clean)
        aligned_corrupted.append(corrupted)
    
    # Stack the aligned tensors to maintain batch dimension
    return torch.stack(aligned_clean), torch.stack(aligned_corrupted)