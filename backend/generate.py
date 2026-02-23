"""
Text generation utilities for the GPT-2 model.
Handles tokenization, inference, and decoding.
"""

import torch
import tiktoken
import os
import json
import numpy as np

# Try tensorflow import for weight loading
try:
    import tensorflow as tf
except ImportError:
    tf = None


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """Load GPT-2 parameters from a TensorFlow checkpoint."""
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]  # Skip 'model/' prefix

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def assign_weights(model, params):
    """Assign loaded TF weights to the PyTorch GPT model."""
    # Token and position embeddings
    model.tok_emb.weight = torch.nn.Parameter(torch.tensor(params["wte"]))
    model.pos_emb.weight = torch.nn.Parameter(torch.tensor(params["wpe"]))

    # Transformer blocks
    for i, block in enumerate(model.trf_blocks):
        block_params = params["blocks"][i]

        # Attention weights - TF checkpoint stores Q, K, V as a single concatenated weight
        attn = block_params["attn"]

        # c_attn contains the combined QKV weights
        qkv_w = torch.tensor(attn["c_attn"]["w"])  # (emb_dim, 3*emb_dim)
        qkv_b = torch.tensor(attn["c_attn"]["b"])  # (3*emb_dim,)

        d_out = model.tok_emb.weight.shape[1]

        # Split into Q, K, V
        block.attention.w_query.weight = torch.nn.Parameter(qkv_w[:, :d_out].T)
        block.attention.w_key.weight = torch.nn.Parameter(qkv_w[:, d_out:2*d_out].T)
        block.attention.w_value.weight = torch.nn.Parameter(qkv_w[:, 2*d_out:].T)

        block.attention.w_query.bias = torch.nn.Parameter(qkv_b[:d_out])
        block.attention.w_key.bias = torch.nn.Parameter(qkv_b[d_out:2*d_out])
        block.attention.w_value.bias = torch.nn.Parameter(qkv_b[2*d_out:])

        # Output projection
        block.attention.out_proj.weight = torch.nn.Parameter(
            torch.tensor(attn["c_proj"]["w"]).T
        )
        block.attention.out_proj.bias = torch.nn.Parameter(
            torch.tensor(attn["c_proj"]["b"])
        )

        # LayerNorm 1
        block.norm1.scale = torch.nn.Parameter(
            torch.tensor(block_params["ln_1"]["g"])
        )
        block.norm1.shift = torch.nn.Parameter(
            torch.tensor(block_params["ln_1"]["b"])
        )

        # Feed-forward
        ff = block_params["mlp"]
        block.ff.layer[0].weight = torch.nn.Parameter(
            torch.tensor(ff["c_fc"]["w"]).T
        )
        block.ff.layer[0].bias = torch.nn.Parameter(
            torch.tensor(ff["c_fc"]["b"])
        )
        block.ff.layer[2].weight = torch.nn.Parameter(
            torch.tensor(ff["c_proj"]["w"]).T
        )
        block.ff.layer[2].bias = torch.nn.Parameter(
            torch.tensor(ff["c_proj"]["b"])
        )

        # LayerNorm 2
        block.norm2.scale = torch.nn.Parameter(
            torch.tensor(block_params["ln_2"]["g"])
        )
        block.norm2.shift = torch.nn.Parameter(
            torch.tensor(block_params["ln_2"]["b"])
        )

    # Final layer norm
    model.final_norm.scale = torch.nn.Parameter(torch.tensor(params["g"]))
    model.final_norm.shift = torch.nn.Parameter(torch.tensor(params["b"]))

    # Output head (weight tied with token embeddings in GPT-2)
    model.out_head.weight = torch.nn.Parameter(torch.tensor(params["wte"]))


def load_model(models_dir="gpt_models", model_size="124M"):
    """
    Load the GPT-2 model with pretrained weights.
    Returns the model and tokenizer.
    """
    from GPTmodel import GPTModel, GPT_Config_124m

    model_dir = os.path.join(models_dir, model_size)

    # Load hparams
    hparams_path = os.path.join(model_dir, "hparams.json")
    with open(hparams_path, "r") as f:
        settings = json.load(f)

    # Load TF checkpoint params
    if tf is None:
        raise ImportError(
            "tensorflow is required to load GPT-2 weights. "
            "Install with: pip install tensorflow"
        )

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if tf_ckpt_path is None:
        raise FileNotFoundError(
            f"No TensorFlow checkpoint found in {model_dir}"
        )

    print(f"Loading weights from: {tf_ckpt_path}")
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    # Update config to allow biases (pretrained model uses them)
    config = GPT_Config_124m.copy()
    config["qkv_bias"] = True

    # Create model and load weights
    model = GPTModel(config)
    assign_weights(model, params)
    model.eval()  # Set to evaluation mode

    # Create tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"Model loaded successfully: GPT-2 {model_size}")
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cpu",
) -> str:
    """
    Generate text from a prompt using the loaded GPT-2 model.

    Args:
        model: The GPT model
        tokenizer: tiktoken tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter (0 = greedy)
        device: Device to run inference on

    Returns:
        Generated text (continuation only, without the prompt)
    """
    model.eval()
    model.to(device)

    # Tokenize the prompt
    token_ids = tokenizer.encode(prompt)
    prompt_len = len(token_ids)
    token_ids = torch.tensor([token_ids], device=device)  # Add batch dimension

    # End-of-text token ID for GPT-2
    EOT_TOKEN = 50256

    # Generate tokens one at a time
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to context length (1024 for GPT-2)
            context = token_ids[:, -1024:]

            # Forward pass
            logits = model(context)

            # Get logits for the last token
            logits = logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    min_top_k = top_k_values[:, -1].unsqueeze(-1)
                    logits = torch.where(
                        logits < min_top_k,
                        torch.full_like(logits, float('-inf')),
                        logits
                    )

                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append the new token
            token_ids = torch.cat([token_ids, next_token], dim=1)

            # Stop if we generate the end-of-text token
            if next_token.item() == EOT_TOKEN:
                break

    # Decode only the generated tokens (not the prompt)
    generated_ids = token_ids[0, prompt_len:].tolist()
    # Remove EOT token if present at end
    if generated_ids and generated_ids[-1] == EOT_TOKEN:
        generated_ids = generated_ids[:-1]

    return tokenizer.decode(generated_ids)
