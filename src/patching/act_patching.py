from typing import Optional, Tuple
import torch
import matplotlib.pyplot as plt
import einops

def plot_top_neurons_by_firing_intensity_diff(
    clean_tokens: torch.Tensor,
    model, 
    tokenizer,
    clean_cache: dict = None,
    corr_cache: dict = None,
    layer: Optional[str] = None,
    neuron: Optional[Tuple[int, int]] = None,
    prompt_idx: int = 1,  # New keyword argument to select the prompt index from the cache.
    plot: bool = False,       # Plot raw neuron activation for a single neuron (if neuron is provided)
    plot_diff: bool = False,    # Plot the difference between main and corr activations (requires corr_cache)
    threshold: Optional[float] = -1.0  # Only keep neurons whose difference is below this value
) -> None:
    """
    Projects a component's activations onto the vocabulary and optionally compares two prompts.
    
    In addition to performing the vocabulary projection on the clean prompt, if a corr_cache is provided,
    the function also uses the alternative (corr) activation to compute the difference in raw neuron firing intensity 
    for every neuron in the selected MLP. If a threshold is provided (e.g. -1), then only neurons whose difference 
    (across token positions) is below that threshold are retained, and a heatmap is produced showing the clean prompt's 
    firing rate for this minimal subset of neurons.
    
    A new kwarg, `prompt_idx`, is added to select which prompt (from the cached batch) to use for plotting.
    
    Args:
        clean_tokens (torch.Tensor): Tokenized clean prompt(s); shape should include a batch dimension.
        model: The pretrained HookedTransformer model.
        tokenizer: The corresponding tokenizer.
        clean_cache (dict): A cache (dictionary) from a forward pass on the clean prompt.
        corr_cache (dict): A cache (dictionary) from a forward pass on the corr prompt.
        layer (Optional[str]): The layer index (as a string) for the MLP to inspect.
        neuron (Optional[Tuple[int, int]]): If provided, a tuple (layer, neuron_index) to extract a single neuron.
                                            **Note:** When using corr_cache for full MLP comparison, do not specify a single neuron.
        prompt_idx (int): Index of the prompt (in the batch dimension) to use for plotting. Default is 1.
        plot (bool): If True and if a single neuron is selected (and corr_cache is not provided), plots its firing intensity.
        plot_diff (bool): If True and if corr_cache is provided (and threshold is not provided), plots a heatmap of 
                          the difference in neuron activations.
        threshold (Optional[float]): If provided (e.g. -1.0), only neurons whose difference (clean - corr) is below 
                                     this value are kept and plotted.
    
    Returns:
        None: This function displays a figure showing the firing rates of the neurons.
    """
    model.reset_hooks()
    
    # Define the module name for the MLP output at the specified layer.
    module_name = f"blocks.{layer}.hook_mlp_out"
    
    if module_name not in clean_cache:
        raise ValueError(f"Activation {module_name} not found in the model (clean cache).")
    activation = clean_cache[module_name]  # Raw activation from the clean prompt

    # === If a corr_cache is provided, compare the activations ===
    if corr_cache is not None:
        if module_name not in corr_cache:
            raise ValueError("Activation for corr_cache not found in the model.")
        activation_corr = corr_cache[module_name]

        # Compute the difference (clean minus corr) for every neuron at each token position.
        diff_activation = activation - activation_corr  # shape: (batch, seq_length, hidden_dim)

        if threshold is not None:        
            # Filter neuron indices where the difference is below the threshold.
            filtered_indices = torch.unique((diff_activation < threshold).nonzero(as_tuple=True)[2])
            if filtered_indices.numel() == 0:
                print("No neurons found below the threshold.")
            else:
                # Use the clean activation for the selected neurons.
                # Use prompt_idx to index the prompt.
                clean_activation_subset = activation[prompt_idx, :, filtered_indices]  # shape: (seq_length, num_filtered)
                token_positions = list(range(clean_activation_subset.shape[0]))
                
                plt.figure(figsize=(12, 6))
                for i, neuron_idx in enumerate(filtered_indices):
                    print(f"Neuron index {neuron_idx} has a diff below {threshold}.")
                    neuron_idx_scalar = neuron_idx.item()
                    neuron_activation = clean_activation_subset[:, i].detach().cpu().numpy()
                    plt.plot(token_positions, neuron_activation, marker='o', linestyle='-', 
                             label=f"Neuron {neuron_idx_scalar}")
                
                plt.xlabel("Token Position")
                plt.ylabel("Clean Activation")
                plt.title(f"Clean Prompt Firing Rates for Neurons in MLP {layer} with Diff < {threshold}")
                
                # Label the x-axis using the tokens from the prompt specified by prompt_idx.
                token_labels = [tokenizer.decode([tok]) for tok in clean_tokens[prompt_idx]]
                plt.xticks(token_positions, token_labels, rotation=45)
                
                plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(False)
                plt.tight_layout()
                plt.show()


def plot_neuron_activation(
    model, clean_cache, tokenizer, clean_tokens, 
    layer, neuron_idx, prompt_idx=0, target_word=" night"
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
    """
    model.reset_hooks()
    
    # Define the module name for the MLP output at the specified layer.
    module_name = f"blocks.{layer}.hook_mlp_out"
    
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
    plt.show()

