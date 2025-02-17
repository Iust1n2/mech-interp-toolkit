import torch
import torch.nn.functional as F
from functools import partial
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm

def compute_succession_score(tl_model, tokenizer, prompts):
    succession_score = torch.zeros((tl_model.cfg.n_layers, tl_model.cfg.n_heads), device=tl_model.cfg.device)
    proportion_in_topk = {}
    succession_results = {}
    act_cache = {}
    
    # Hook function to capture activations
    def hook_fn(activation: torch.Tensor, hook: HookPoint, name: str = "activation"):
        """Stores activations in hook context."""
        act_cache[name] = activation
        return activation
    # Forward hooks to capture relevant activations
    fwd_hooks = [
    ("hook_embed", partial(hook_fn, name="Embed")),
    ("blocks.0.hook_resid_post", partial(hook_fn, name="MLP0")),
    ]

    tl_model.run_with_hooks(prompts, fwd_hooks=fwd_hooks)

    # Retrieve activations
    WE_cached = act_cache["Embed"]  # Shape: [seq_len, d_model]
    WE = tl_model.embed  # Shape: [seq_len, d_model]
    MLP0 = act_cache["MLP0"]  # Shape: [batch, pos, d_model]
    WU = tl_model.unembed
    ln_final = tl_model.ln_final
    
    tokens = tl_model.to_tokens(prompts, prepend_bos=False)
    total_tokens = tokens.shape[1]

    # Loop over all (layer, head) pairs
    for layer in tqdm(range(tl_model.cfg.n_layers)):
        for head in range(tl_model.cfg.n_heads):
            # print(f"Computing Succession Score for Layer {layer}, Head {head}...")
            token_success_count = 0  # Count tokens where the condition is met
            for token_idx in range(total_tokens - 1):  # Skip last token
                
                # Retrieve token embedding and successor ID
                token_emb = WE_cached[:, token_idx]  # Shape: [d_model]
                successor_id = tokens[0, token_idx + 1].item()  # Successor token ID
                
                # # Compute the residual stream after MLP0
                # MLP0_rearranged = einops.rearrange(MLP0, "b p d -> (b p) d")
                # MLP0_rearranged_single = MLP0_rearranged[token_idx, :]  # Shape: [d_model]
                resid_after_mlp1 = token_emb + MLP0
                
                # Compute W_OV and residual output
                W_OV = tl_model.W_V[layer, head] @ tl_model.W_O[layer, head]  
                assert W_OV.shape == (tl_model.cfg.d_model, tl_model.cfg.d_model)
                
                # Residual stream after applying OV                
                resid_after_ov = resid_after_mlp1 @ W_OV  
                
                # Compute effective circuit logits
                logits = WU(ln_final(resid_after_ov)).squeeze()  # Shape: [vocab_size]

                # probs = torch.softmax(logits, dim=-1)
                # cumulative_probs = probs.sort(descending=True).values.cumsum(dim=-1)
                # k_dynamic = (cumulative_probs < 0.95).sum().item()  # k where 95% of the probability mass is included
                # k = min(k_dynamic, 100)  # Cap k at 100 for efficiency

                # Compute top-k logits and their indices
                top_k_values, topk_logits = torch.topk(logits, dim=-1, k=100) 

                # Find the position of the successor token in the top-k logits
                successor_pos_in_topk = None
                if successor_id in topk_logits:
                    matches = (topk_logits == successor_id).nonzero(as_tuple=True)
                    if matches[0].numel() > 0:  # Ensure there is at least one match
                        successor_pos_in_topk = matches[0][0].item()  # Use the first match

                # Check positions of all other subsequent tokens in the sequence
                is_successor_before_others = True
                for other_idx in range(token_idx + 2, tokens.shape[1]):  # Loop through subsequent tokens
                    other_token_id = tokens[0, other_idx].item()

                    # Check if the other token is in the top-k
                    other_token_pos_in_topk = None
                    if other_token_id in topk_logits:
                        matches = (topk_logits == other_token_id).nonzero(as_tuple=True)
                        if matches[0].numel() > 0:  # Ensure there is at least one match
                            other_token_pos_in_topk = matches[0][0].item()  # Use the first match

                    # If the successor is not before this token, fail the condition
                    if successor_pos_in_topk is None or (other_token_pos_in_topk is not None and successor_pos_in_topk > other_token_pos_in_topk):
                        is_successor_before_others = False
                        break

                if is_successor_before_others:
                    token_success_count += 1

                # # Debugging Output
                # current_token = tokenizer.decode([tokens[0, token_idx].item()], clean_up_tokenization_spaces=True).strip()
                # successor_token = tokenizer.decode([successor_id], clean_up_tokenization_spaces=True).strip()
                # print(f"Current Token: '{current_token}', Successor Token: '{successor_token}'")
                # print(f"Successor Position in Top-k: {successor_pos_in_topk}")
                # if not is_successor_before_others:
                #     print("The successor appears after another token in the top-k.")

            # Compute the succession score
            succession_condition_met = token_success_count > (total_tokens / 2)
            proportion_in_topk[(layer, head)] = token_success_count / total_tokens
            if succession_condition_met:
                succession_score[layer, head] = 1

    # Save results
    succession_results = {
        "succession_scores": succession_score,
        "proportion_in_topk": proportion_in_topk,
    }

    return succession_results

@torch.no_grad()
def run_model(model, prompts, fwd_hooks, prepend_bos=True, split_qkv=True):

    model.set_use_attn_result(True)
    if split_qkv:
        model.set_use_split_qkv_input(True)
    else:
        model.set_use_split_qkv_input(False)

    model.reset_hooks()
    #print("PRO:", prompts)
    logits = model.run_with_hooks(
        prompts,
        fwd_hooks=fwd_hooks,
        prepend_bos=prepend_bos,
    )[:, -1, :] # (b, vocab_size)
    model.reset_hooks()

    probs = F.softmax(logits, dim=-1) # (b, vocab_size)

    return (logits, probs)

def get_top_k_strings(model, logits, k, probs=None, use_br=True):
    # logits: (b, vocab_size)

    logit_tops, top_idxs = torch.topk(logits, k, dim=-1) # each (b, k)

    prob_tops = None
    if probs is not None:
        prob_tops, _ = torch.topk(probs, k, dim=-1)

    top_toks = ["" for _ in range(logits.size(0))]
    
    for i in range(logits.size(0)):
        for tok_id in top_idxs[i]:
            top_toks[i] += f"{model.tokenizer.decode([tok_id])}"

            if use_br:
                top_toks[i] += "<br />"
            else:
                top_toks[i] += ","

    return top_toks, (logit_tops, prob_tops)

def hook_edit_pattern(layer, head, seq_idx):

    def edit_pattern(tensor, hook, head, seq_idx):
        # tensor: (b, n_heads, seq_len, seq_len)

        tensor[:, head, -1, :seq_idx] = 0.0
        tensor[:, head, -1, seq_idx+1:] = 0.0
        tensor[:, head, -1, seq_idx] = 1.0
        
        return tensor

    return (f"blocks.{layer}.attn.hook_pattern", partial(edit_pattern, head=head, seq_idx=seq_idx))

def hook_save_layer_input(layer, save_arr):

    def save_layer_input(tensor, hook, save_arr):
        # tensor: (b, seq_len, d_model)
        save_arr.append(tensor.clone())
        return tensor

    return (f"blocks.{layer}.hook_resid_pre", partial(save_layer_input, save_arr=save_arr))

def hook_save_head_output(model, layer, head, save_arr, include_bias):

    def save_head_output(tensor, hook, layer, head, save_arr, include_bias):
        # tensor: (b, seq_len, n_heads, d_model)
        if include_bias:
            save_arr.append((tensor[:, :, head, :] + model.blocks[layer].attn.b_O).clone())
        else:
            save_arr.append(tensor[:, :, head, :].clone())
        return tensor

    return (f"blocks.{layer}.attn.hook_result", partial(save_head_output, layer=layer, head=head, save_arr=save_arr, include_bias=include_bias))

def hook_save_attention_output(model, layer, save_arr, include_bias):

    def save_attention_output(tensor, hook, layer, head, save_arr, include_bias):
        # tensor: (b, seq_len,  d_model)
        if include_bias:
            save_arr.append((tensor[:, :, :] + model.blocks[layer].attn.b_O).clone())
        else:
            save_arr.append(tensor[:, :, :].clone())
        return tensor

    return (f"blocks.{layer}.hook_attn_out", partial(save_attention_output, layer=layer, save_arr=save_arr, include_bias=include_bias))

def hook_save_layer_output(layer, save_arr):

    def save_layer_output(tensor, hook, save_arr):
        # tensor: (b, seq_len, d_model)
        save_arr.append(tensor.clone())
        return tensor
    
    return (f"blocks.{layer}.hook_resid_post", partial(save_layer_output, save_arr=save_arr))

def hook_save_head_pattern(layer, head, save_arr):

    def save_head_pattern(tensor, head, save_arr):
        # tensor: (b, n_heads, seq_len, seq_len)
        save_arr.append(tensor[:, head, :, :].clone())
        return tensor

    return (f"blocks.{layer}.attn.hook_pattern", partial(save_head_pattern, head=head, save_arr=save_arr))

def hook_ablate_head(layer, head, means):
    # means: (b, n_layers, 1, seq_len, n_heads, d_model)
    # Replaces output of (layer, head) with means[:, layer, 0, :, head, :]

    def ablate_head(tensor, hook, layer, head, means):
        # tensor: (b, seq_len, n_heads, d_model)
        if means is None:
            tensor[:, :, head, :] = 0.0
        
        else:
            tensor[:, :, head, :] = means[:, layer, 0, :, head, :]

        return tensor

    return (f"blocks.{layer}.attn.hook_result", partial(ablate_head, layer=layer, head=head, means=means))


def hook_ablate_block(layer, means): 
    # means: (b, n_layers, 1, seq_len, d_model)
    # Replaces output of (layer) with means[:, layer, 0, :]
    def ablate_block(tensor, hook, layer, means):
        # tensor: (b, seq_len, d_model)
        if means is None:
            tensor = 0.0
        else:
            tensor[:, layer, :, :] = 0.

        return tensor

    return (f"blocks.{layer}.hook_attn_out", partial(ablate_block, layer=layer, means=means))


def hook_set_layer_input(layer, new_input):
    # new_input: (b, seq_len, d_model)

    def set_layer_input(tensor, hook, new_input):
        # tensor: (b, seq_len, d_model)
        if isinstance(new_input, list):
            new_input = new_input[0]
        tensor.copy_(new_input)
        return tensor

    return (f"blocks.{layer}.hook_resid_pre", partial(set_layer_input, new_input=new_input))

def hook_set_layer_output(layer, new_output):
    # new_output: (b, seq_len, d_model)

    def set_layer_output(tensor, hook, new_output):
        # tensor: (b, seq_len, d_model)
        if isinstance(new_output, list):
            new_output = new_output[0]
        tensor.copy_(new_output)
        return tensor

    return (f"blocks.{layer}.hook_resid_post", partial(set_layer_output, new_output=new_output))

