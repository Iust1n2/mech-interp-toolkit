from typing import List
import plotly.express as px
import numpy as np
from fancy_einsum import einsum
import einops
import torch
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float
from transformer_lens import HookedTransformer, ActivationCache

import sys
sys.path.append("../")
from succession_utils import run_model, hook_save_head_output, get_top_k_strings

def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    per_prompt: bool = False
) -> Float[Tensor, "*batch"]:
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()



def residual_stack_to_logit_diff(
            residual_stack: Float[Tensor, "components batch d_model"],
            act_cache: ActivationCache,
            logit_diff_directions: Float[Tensor, "batch d_model"],
            prompts: List[str]
        ) -> float:
            # Applies layer normalization scaling to the residual stack (division by norm) returns residual_stack / scale 
            # which is a global property of the Transformer
            scaled_residual_stack = act_cache.apply_ln_to_stack(
                residual_stack, layer=-1, pos_slice=-1
            )
            return einsum(
                "... batch d_model, batch d_model -> ...",
                scaled_residual_stack,
                logit_diff_directions,
            ) / len(prompts)

def logit_lens(
            tl_model: HookedTransformer, 
            prompts: List[str], 
            act_cache: ActivationCache, 
            answer_tokens_ids: List[List[int]], 
            decomposition: str
            ) -> float:
        # Returns a stack of mapped answer tokens (correct and wrong) to a tensor with the unembedding vector for those tokens 
        # (W_U[:, token] of shape d_model)
        answer_residual_directions = tl_model.tokens_to_residual_directions(answer_tokens_ids) # shape (batch_size, 2, d_model), where the 2nd dim is the correct and wrong answer
        print("Answer residual directions shape:", answer_residual_directions.shape)
        
        # Calculate the difference between the logits of the two answers
        logit_diff_directions = (
            answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
        )
        # print("Logit difference directions shape:", logit_diff_directions.shape)

        # Accumulate the residuals for the last layer
        accumulated_residual, labels = act_cache.accumulated_resid(
            layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
        )
        if decomposition == "residual":
            logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, act_cache, logit_diff_directions, prompts)
        elif decomposition == "layer_blocks":
            # Decompose the residual stream input to layer L into the sum of the outputs of previous compoments (plus W_E and W_pos_emb)
            per_layer_residual, labels = act_cache.decompose_resid(
            layer=-1, pos_slice=-1, return_labels=True
            )
            logit_lens_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, act_cache, logit_diff_directions, prompts)
        
        elif decomposition == "attention_heads": 
            per_head_residual, labels = act_cache.stack_head_results(
                layer=-1, pos_slice=-1, return_labels=True
            )
            print(per_head_residual.shape)
            logit_lens_logit_diffs = residual_stack_to_logit_diff(per_head_residual, act_cache, logit_diff_directions, prompts)
            print(logit_lens_logit_diffs.shape)
            logit_lens_logit_diffs = einops.rearrange(
                logit_lens_logit_diffs,
                "(layer head_index) -> layer head_index",
                layer=tl_model.cfg.n_layers,
                head_index=tl_model.cfg.n_heads,
            )
        return logit_lens_logit_diffs, labels


def visual_logit_lens(tl_model, prompts, titles):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl_model = tl_model.to(device)

    # Tokenize prompts and determine correct IDs
    tokens = tl_model.to_tokens(prompts, prepend_bos=True).to(device)
    correct_ids = tokens[:, 1:]  # Shifted tokens (t -> t+1)
    tokens = tokens[:, :-1]  # Adjust tokens to align with correct IDs

    assert tokens.shape == correct_ids.shape, f"Shape mismatch: {tokens.shape} vs {correct_ids.shape}"

    # Sanity check for token IDs
    vocab_size = tl_model.cfg.d_vocab
    assert correct_ids.max() < vocab_size, "Token ID exceeds vocabulary size"
    assert correct_ids.min() >= 0, "Token ID is negative"

    n_layers = tl_model.cfg.n_layers
    n_heads = tl_model.cfg.n_heads
    unembed = torch.nn.Sequential(
        tl_model.ln_final,
        tl_model.unembed,
    )

    str_matrix = np.empty((tokens.shape[1], n_layers, n_heads), dtype=object)
    prob_matrices = torch.zeros(tokens.shape[1], n_layers, n_heads, device=device)

    out_arrs = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    fwd_hooks = [
        hook_save_head_output(tl_model, layer, head, out_arrs[layer][head], include_bias=False)
        for layer in range(n_layers) for head in range(n_heads)
    ]
    # Forward pass with hooks
    _ = run_model(tl_model, tokens, fwd_hooks, split_qkv=False)  

    for layer in range(n_layers):
        for head in range(n_heads):
            out = out_arrs[layer][head][0]

            logits = unembed(out)[:, -1, :]
            probs = F.softmax(logits, dim=-1)  # (b, vocab_size)

            # Get top-k strings for visualization
            top_toks, _ = get_top_k_strings(tl_model, logits, k=3, use_br=True)

            str_matrix[:, layer, head] = top_toks
            prob_matrices[:, layer, head] = probs[torch.arange(tokens.shape[0]), correct_ids.flatten()]

    # Visualize for each token -> next token pair
    for token_idx in range(tokens.shape[1]):
        head_fig = px.imshow(
            prob_matrices[token_idx].detach().cpu().T,
            range_color=(0, 1),
            origin="lower",
            height=1100,
            width=1300,
            labels=dict(x="Layers", y="Heads"),
        )

        head_fig.update_traces(
            text=str_matrix[token_idx].T,
            texttemplate="%{text}",
            textfont=dict(
                family="Courier New, monospace",
                size=12,
                color="white"
            )
        )

        head_fig.update_layout(
            title_text=titles[token_idx],
            title_x=0.5
        )

        head_fig.show()