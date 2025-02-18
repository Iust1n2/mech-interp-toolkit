import numpy as np
import torch
import plotly.express as px
import matplotlib.pyplot as plt
from functools import partial
from typing import Optional
from transformer_lens.hook_points import HookPoint
from data.ioi_dataset import IOIDataset

def parse_activation_identifier(identifier: str) -> str:
    """
    Parses a simplified activation identifier and maps it to the corresponding module name.

    Supported formats:
    - Residual stream: L9.Resid_Pre, L9.Resid_Mid, L9.Resid_Post
    - MLP: L9.MLP_In, L9.MLP_Out
    - Attention output: L9.H9.Output
    - Attention heads: L9.H9.Q, L9.H9.K, L9.H9.V, L9.H9.Z

    Args:
        identifier (str): Simplified activation identifier (e.g., 'L9.Resid_Pre').

    Returns:
        str: Corresponding module name in the model.
    """
    parts = identifier.split('.')
    if len(parts) < 2:
        raise ValueError(f"Invalid activation identifier format: {identifier}")

    layer_str, component = parts[0], '.'.join(parts[1:])

    # Validate and convert layer index
    if not (layer_str.startswith('L') and layer_str[1:].isdigit()):
        raise ValueError("Invalid layer specification. Expected format 'L{layer_number}'.")
    layer = int(layer_str[1:])

    # Map components to model module names
    if component.startswith("Resid_"):
        # Residual stream components
        resid_map = {
            "Resid_Pre": "hook_resid_pre",
            "Resid_Mid": "hook_resid_mid",
            "Resid_Post": "hook_resid_post",
        }
        if component not in resid_map:
            raise ValueError(f"Invalid residual stream component: {component}")
        return f"blocks.{layer}.{resid_map[component]}"

    elif component.startswith("MLP_"):
        # MLP components
        mlp_map = {
            "MLP_In": "hook_mlp_in",
            "MLP_Out": "hook_mlp_out",
        }
        if component not in mlp_map:
            raise ValueError(f"Invalid MLP component: {component}")
        return f"blocks.{layer}.{mlp_map[component]}"

    elif component.startswith("H"):
        # Attention heads components
        head_parts = component.split('.')
        if len(head_parts) != 2:
            raise ValueError(f"Invalid attention head specification: {component}")

        head_str, subcomponent = head_parts
        if not (head_str.startswith('H') and head_str[1:].isdigit()):
            raise ValueError("Invalid attention head specification. Expected format 'H{head_number}'.")
        head = int(head_str[1:])

        attn_map = {
            "Q": "hook_q",
            "K": "hook_k",
            "V": "hook_v",
            "Z": "hook_z",
        }
        if subcomponent not in attn_map:
            raise ValueError(f"Invalid attention head subcomponent: {subcomponent}")
        return f"blocks.{layer}.attn.{attn_map[subcomponent]}"
    elif component == "Attn_Output":
        return f"blocks.{layer}.hook_attn_out"
    else:
        raise ValueError(f"Unknown component: {component}")

def format_percentage_position(value, tmp_desc, total_length):
        tmp_desc_np = tmp_desc.cpu().detach().numpy()
        position = len(np.where(tmp_desc_np >= value.cpu().detach().numpy())[0])
        percentage_position = position / total_length
        formatted_percentage = f'{percentage_position:.4%}'
        return formatted_percentage, percentage_position


def show_attention_patterns(
    model,
    heads,
    prompts,
    precomputed_cache=None,
    mode="val",
    title_suffix="",
    return_fig=False,
    return_mtx=False,
):
    """
    Visualizes attention patterns for the specified heads in the model.

    Args:

    model (torch.nn.Module): Model to visualize.
    heads (List[Tuple[int, int]]): List of tuples specifying the layer and head indices to visualize.
    ioi_dataset (IOIDataset): Dataset containing the input sequences.
    precomputed_cache (Dict[str, torch.Tensor], optional): Precomputed activations cache.
    mode (str): Visualization mode ('pattern', 'val', 'scores').
    title_suffix (str): Suffix to append to the plot title.
    return_fig (bool): Whether to return the plotly figure.
    return_mtx (bool): Whether to return the attention matrices.

    Returns:
    - If return_fig=True and return_mtx=False, returns the plotly figure.
    - If return_fig=False and return_mtx=True, returns the attention matrices.
    - If return_fig=False and return_mtx=False, displays the attention patterns.

    Info:
    - 'pattern': Visualizes the attention patterns post-softmax, or attention probabilities,
    - 'val': Visualizes the value-weighted attention patterns,
    - 'scores': Visualizes the attention scores pre-softmax, 

    Note: 
    ! More about the types of activations on:  https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.ActivationCache.html
    """
    assert mode in [
        "pattern",
        "val",
        "scores",
    ]  # value weighted attention or attn for attention probas
    
    assert len(heads) == 1 or not (return_fig or return_mtx)

    for (layer, head) in heads:
        cache = {}
        good_names = []
        good_names.append(f"blocks.{layer}.attn.hook_v")
        # if mode == "pattern":
        good_names.append(f"blocks.{layer}.attn.hook_pattern")
        # elif mode == "scores":
        good_names.append(f"blocks.{layer}.attn.hook_attn_scores")
        # if hasattr(model.cfg, "n_key_value_heads"):
        good_names.append(f"blocks.{layer}.attn.hook_z")
        # else:
        if precomputed_cache is None:
            cache = {}
            def hook_fn(activation: torch.Tensor, hook: HookPoint, name: str = "activation"):
                '''Stores activations in hook context.'''
                cache[name] = activation
                return activation

            fwd_hooks = [
                (good_names[0], partial(hook_fn, name=good_names[0])), # hook_v
                (good_names[1], partial(hook_fn, name=good_names[1])), # hook_pattern
                (good_names[2], partial(hook_fn, name=good_names[2])), # hook_attn_scores
                (good_names[3], partial(hook_fn, name=good_names[3])), # hook_z
            ]
            model.run_with_hooks(prompts, fwd_hooks=fwd_hooks)
        else:
            cache = precomputed_cache
         
        attn_results = torch.zeros(
            size=(len(prompts), len(prompts[0]), len(prompts[0]))
        )
        attn_results += -20

        toks = model.tokenizer(prompts)["input_ids"]
        current_length = len(toks)
        words = [model.tokenizer.decode([tok]) for tok in toks]
        attn_pattern = cache[good_names[1]].detach().cpu()[:, head, :, :].squeeze(0)
        attn_scores = cache[good_names[2]].detach().cpu()[:, head, :, :].squeeze(0)
        if mode == "val":
            if not hasattr(model.cfg, "n_key_value_heads"):
                vals = cache[good_names[0]].detach().cpu()[:, :, head, :].norm(dim=-1).squeeze(0)
                # print(f"vals shape: {vals.shape}")
                # print(f"attn_pattern shape: {attn_pattern.shape}")
                cont = torch.einsum("ab,b->ab", attn_pattern, vals)        
            else:
                cont = cache[good_names[3]].detach().cpu()[0, :, head, :]
                print(cont.shape)

        fig = px.imshow(
            attn_pattern if mode == "pattern" else attn_scores if mode == "scores" else cont,
            title=f"{layer}.{head} Attention" + title_suffix,
            color_continuous_midpoint=0,
            color_continuous_scale="RdBu",
            labels={"y": "Queries", "x": "Keys"},
            height=600,
        )

        fig.update_layout(
            xaxis={
            "side": "top",
            "ticktext": words,
            "tickvals": list(range(len(words))),
            "tickfont": dict(size=15),
            },
            yaxis={
            "ticktext": words,
            "tickvals": list(range(len(words))),
            "tickfont": dict(size=15),
            },
            width=800, 
            height=650,
        )
        if return_fig and not return_mtx:
            return fig
        elif return_mtx and not return_fig:
            if mode == "val":
                return cont
            elif mode == "pattern":
                attn_results[:, :current_length, :current_length] = (
                    attn_pattern[:current_length, :current_length].clone().cpu()
                )
            elif mode == "scores":
                attn_results[:, :current_length, :current_length] = (
                    attn_scores[:current_length, :current_length].clone().cpu()
                )
        else:
            fig.show()

        if return_fig and not return_mtx:
            return fig
        elif return_mtx and not return_fig:
            return attn_results
        
def plot_neuron_activation(
    model, clean_cache, tokenizer, clean_tokens, 
    layer, neuron_idx, prompt_idx=0, target_word=" night", save=False
):
    """
    Extracts the activations of a specific neuron in the clean distribution 
    and checks its response to a given token (e.g., " night").

    Args:
        model: TransformerLens model.
        clean_cache (dict): Activation cache from the clean prompt.
        tokenizer: Tokenizer corresponding to the model.
        clean_tokens (torch.Tensor): Tokenized clean prompts.
        layer (int): MLP layer index to analyze.
        neuron_idx (int): Index of the MLP neuron to examine.
        prompt_idx (int): Index of the prompt to analyze.
        target_word (str): Token to check for strong activation.
        save (bool): Whether to save the plot as an image.
    """
    model.reset_hooks()
    
    # Define the module name for the MLP output at the specified layer.
    module_name = f"blocks.{layer}.mlp.hook_post"
    
    if module_name not in clean_cache:
        raise ValueError(f"Activation {module_name} not found in the model (clean cache).")
    
    activation = clean_cache[module_name]  # Shape: (batch, seq_length, hidden_dim)

    # Extract activations for the specific neuron
    neuron_activation = activation[prompt_idx, :, neuron_idx]  # Shape: (seq_length,)

    # Decode tokens for x-axis labels
    token_labels = [tokenizer.decode([tok]) for tok in clean_tokens[prompt_idx]]

    # Find the position of the target token (e.g., "night")
    try:
        target_idx = token_labels.index(target_word)
        print(f"Target word '{target_word}' found at position {target_idx}. Activation: {neuron_activation[target_idx].item():.4f}")
    except ValueError:
        print(f"Target word '{target_word}' not found in the tokenized prompt.")
        target_idx = None

    # Plot neuron activation across all tokens
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(token_labels)), neuron_activation.cpu().numpy(), marker="o", linestyle="-", color="blue", label=f"Neuron {neuron_idx}")

    if target_idx is not None:
        plt.scatter([target_idx], [neuron_activation[target_idx].item()], color="red", s=100, zorder=3, label=f'"{target_word}" token')

    plt.xticks(range(len(token_labels)), token_labels, rotation=45, ha="right", fontsize=10)
    plt.xlabel("Token Position")
    plt.ylabel("Activation Value")
    plt.title(f"Neuron {neuron_idx} Activation Across Tokens (MLP Layer {layer})")
    plt.legend()
    plt.grid(False)

    plt.tight_layout()  # Adjust layout before saving
    if save:
        # Create the directory if it doesn't exist
        output_dir = f"figures/mlp/firing/mlp_{layer}/neuron_{neuron_idx}"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"figures/mlp/firing/mlp_{layer}/neuron_{neuron_idx}/prompt_{prompt_idx}.png", bbox_inches='tight')
    plt.show()