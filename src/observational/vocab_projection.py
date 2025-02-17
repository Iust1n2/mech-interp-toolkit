import numpy as np
from typing import List, Optional, Tuple
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.components import Embed
from transformer_lens.hook_points import HookPoint
from torch import Tensor
from transformers import AutoTokenizer
import einops
import torch
from jaxtyping import Float, Int
from functools import partial
import torch.nn.functional as F
from typing import Callable

from src.observational.utils import parse_activation_identifier, format_percentage_position

def vocabulary_projection(
    prompt: str,
    model: HookedTransformer,
    tokenizer,
    method: str = "unembedding",
    top_k: int = 5,
    all_tok_pos: bool = True,
    specific_activation: Optional[str] = None,
    neuron: Optional[Tuple[int, int]] = None,
) -> List[Tuple[str, float]]:
    """
    Projects the vocabulary onto a component's activations in a Transformer using a PyTorch hook.

    Args:
        prompt (str): The input textual prompt.
        model (HookedTransformer): Pretrained Transformer model from TransformerLens.
        tokenizer: Corresponding tokenizer for the model.
        method (str): Method for projection ('unembedding' or 'top-k').
        top_k (int): Number of tokens to return if using top-k.
        all_tok_pos (bool): Whether to project hidden states for all token positions or just the last one.
        specific_activation (Optional[str]): Simplified activation identifier (e.g., 'H9.9.K').
        neuron (Optional[Tuple[int, int]]): Tuple specifying the layer and neuron index to project to.

    Returns:
        List[Tuple[str, float]]: List of tokens and their probabilities.
    """
    model.reset_hooks()
    # Tokenize the prompt
    tokens = tokenizer.encode(prompt, return_tensors='pt')

    # Dictionary to store activations
    act_cache = {}

    # Hook function to capture activations
    def hook_fn(activation: torch.Tensor, hook: HookPoint, name: str = "activation"):
        '''Stores activations in hook context.'''
        act_cache[name] = activation
        return activation

    # Prepare hooks
    module_name = parse_activation_identifier(specific_activation)
    fwd_hooks: List[Tuple[str, callable]] = [
        (module_name, partial(hook_fn, name='specific'))
    ]

    # Run the model with hooks
    model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    # model(tokens)

    # Check if the specific activation was captured
    if 'specific' in act_cache:
        activation = act_cache['specific']
    else:
        raise ValueError(f"Activation {specific_activation} not found in the model.")

    # Handle neuron-specific projection
    if neuron:
        neuron_index = neuron
        if len(activation.shape) != 3:  # Ensure activation is (batch_size, seq_length, hidden_dim)
            raise ValueError(f"Activation shape {activation.shape} is not suitable for neuron projection.")
        # Isolate the specific neuron
        neuron_activation = activation[:, :, neuron_index]  # Shape: (batch_size, seq_length)
        # Expand it to match the hidden size of the model
        activation = neuron_activation.unsqueeze(-1).expand(-1, -1, model.cfg.d_model)  # Shape: (batch_size, seq_length, hidden_dim)

    # Apply layer normalization if using the unembedding method
    if method == "unembedding":
        if len(activation.shape) == 4:  # Shape: (batch_size, seq_length, num_heads, head_dim)
            activation_reshaped = einops.rearrange(activation, 'b s h d -> b s (h d)')
        elif len(activation.shape) == 3:  # Shape: (batch_size, seq_length, hidden_dim)
            activation_reshaped = activation
        # Apply final layer normalization
        activation_ln = model.ln_final(activation_reshaped)

        # Project to vocabulary space using the unembedding matrix
        logits = torch.matmul(activation_ln, model.W_U)  # (batch_size, seq_length, vocab_size)
        probs = F.softmax(logits, dim=-1)  # (batch_size, seq_length, vocab_size)
    
    elif method == "top-k":
        if len(activation.shape) == 3:  # Residual stream activations
            # Apply layer normalization to align with vocabulary space
            activation_ln = model.ln_final(activation)
            # Apply softmax to get probabilities
            probs = F.softmax(activation_ln, dim=-1)
        elif len(activation.shape) == 4:  # Attention head activations
            activation_reshaped = einops.rearrange(activation, 'b s h d -> b s (h d)')
            probs = F.softmax(activation_reshaped, dim=-1)
        else:
            raise ValueError(f"Unexpected activation shape {activation.shape} for top-k.")

    else:
        raise ValueError("Invalid method. Choose 'unembedding' or 'top-k'.")
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

    if not all_tok_pos:
        # Decode top-k tokens for the last token
        top_k_tokens = [tokenizer.decode(idx.item()) for idx in top_k_indices[0, -1, :]]
        # Convert probabilities to a list
        top_k_probabilities = top_k_probs[0, -1, :].tolist()
        # Combine tokens and probabilities
        top_k_results = list(zip(top_k_tokens, top_k_probabilities))
    else:
        # Decode top-k tokens for each token position
        top_k_results = []
        for position in range(probs.shape[1]):  # Iterate over sequence length
            tokens_at_pos = [tokenizer.decode(idx.item()) for idx in top_k_indices[0, position, :]]
            probabilities_at_pos = top_k_probs[0, position, :].tolist()
            top_k_results.append(list(zip(tokens_at_pos, probabilities_at_pos)))

    return top_k_results, activation


def inspect_ov_matrix(model: HookedTransformer, tokenizer: AutoTokenizer, tokens: torch.Tensor, attn_head: Tuple[int, int], pairwise: bool = False):
    """
    Inspects the OV matrix of a model for a specified Attention Head given sequence of tokens.
    Code taken primarily from: https://github.com/guy-dar/embedding-space

    The OV matrix is a low-rank refactored matrix from einsum of W_V and W_O matrices.
    For more details check HookedTransformer.refactored_attn_matrices() method.

    Args:
        model (HookedTransformer): The model to inspect.
        tokenizer (AutoTokenizer): The tokenizer used to encode the tokens.
        tokens (torch.Tensor): The list of tokens to inspect.
        attn_head (Tuple[int, int]): The Attention Head to inspect.
        pairwise (bool): Whether to inspect pairwise token combinations. Default is False.
    
    Returns:
        Prints the token as entries in the OV matrix in descending order. Equivalent to a filtered top-k.
    """ 
    if tokenizer is None:
            try: 
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
            except OSError as e:
                print(f"Error: {e}")
                print("Please provide a valid tokenizer")
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    hidden_dim = model.cfg.d_model  
    head_size = hidden_dim // n_heads

    layer_idx = attn_head[0]
    head_idx = attn_head[1]

    # Using model.OV
    # W_OV_head = model.OV[layer_idx, head_idx, : , :]  # Shape: (hidden_dim, hidden_dim)
    # print(f"W_OV_head shape: {W_OV_head.shape}") # [n_layers, n_heads, d_model, d_model]
    
    # Extract W_V and W_O for the specific head
    print(f"W_V shape: {model.W_V.shape}")
    print(f"W_O shape: {model.W_O.shape}")
    W_V_head = model.W_V[layer_idx, head_idx, :]  
    W_O_head = model.W_O[layer_idx, head_idx] 

    print(f"W_V_head shape: {W_V_head.shape}")
    print(f"W_O_head shape: {W_O_head.shape}")

    # # Compute OV matrix for the specific head
    # W_OV_head = torch.matmul(W_V_head, W_O_head)  # Shape: (hidden_dim, hidden_dim)
    # print(f"W_OV_head shape: {W_OV_head.shape}")

    # Get embeddings and their transpose
    emb = model.embed
    emb_matrix = emb.W_E  # Shape: (d_vocab, d_model)
    emb_inv = emb_matrix.T  # Shape: (d_model, d_vocab)

    # print(f"emb shape: {emb.get_shape()}")
    print(f"emb matrix shape: {emb_matrix.shape}")
    print(f"emb_inv shape: {emb_inv.shape}")

    # Multiply embeddings with the OV matrix
    # tmp = emb_matrix @ W_OV_head @ emb_inv  # Shape: (d_vocab, d_vocab)
    tmp = (emb_matrix @ (W_V_head @ W_O_head) @ emb_inv)

    # Print results
    print(f"tmp shape: {tmp.shape}") # (d_vocab, d_vocab)
    
    tmp_desc = tmp.flatten() # Shape: (d_vocab * d_vocab)
    print(f"tmp_desc shape: {tmp_desc.shape}")

    l = len(tmp_desc)

    if pairwise: 
        for token_1, token_2 in tokens:
            # Encode tokens
            token_1_encoded = tokenizer.encode(token_1)
            token_2_encoded = tokenizer.encode(token_2)

            # Get the matrix value for the token pair
            value = tmp[token_1_encoded, token_2_encoded]

            for v in value: 
                formatted_percentage, _ = format_percentage_position(v, tmp_desc, l)
                print(f"'{token_1}, {token_2}': {formatted_percentage}")
            # Free memory
        del tmp
        torch.cuda.empty_cache()
    else:
        for token in tokens:
            token_encoded = tokenizer.encode(token)
            
            value = tmp[token_encoded].item()
            
            for v in value: 
                formatted_percentage, _ = format_percentage_position(v, tmp_desc, l)
                print(f"'{token}': {formatted_percentage}")
        del tmp
        torch.cuda.empty_cache()

