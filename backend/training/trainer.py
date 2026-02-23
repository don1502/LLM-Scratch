"""
Training utilities for the GPT model.
Contains training loop, loss calculation, evaluation, and text generation helpers.
"""

import torch
import tiktoken


def text_to_token(text, tokenizer):
    """Convert text string to token tensor with batch dimension."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    return encoded_tensor


def token_to_text(token_ids, tokenizer):
    """Convert token tensor back to text string."""
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text(model, idx, max_new_token, context_size):
    """
    Generate text tokens autoregressively.

    Args:
        model: GPT model
        idx: Input token tensor (batch, n_tokens)
        max_new_token: Number of new tokens to generate
        context_size: Maximum context window size

    Returns:
        Extended token tensor with generated tokens appended
    """
    for _ in range(max_new_token):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probab = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probab, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def cal_loss_batch(input_batch, target_batch, model, device):
    """Calculate cross-entropy loss for a single batch."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def cal_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate average loss over a dataloader.

    Args:
        data_loader: DataLoader to evaluate
        model: GPT model
        device: torch device
        num_batches: Optional limit on number of batches to evaluate

    Returns:
        Average loss as float
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate model on both training and validation sets."""
    model.eval()
    with torch.no_grad():
        train_loss = cal_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = cal_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print(model, tokenizer, device, start_context):
    """Generate and print sample text during training."""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_id = generate_text(
            model=model, idx=encoded, max_new_token=50, context_size=context_size
        )
    decoded_text = token_to_text(token_id, tokenizer)
    print(decoded_text.replace("\n", " "))


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epoch,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    """
    Main training loop for GPT model.

    Args:
        model: GPT model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer
        device: torch device
        num_epoch: Number of training epochs
        eval_freq: Evaluate every N global steps
        eval_iter: Number of batches for evaluation
        start_context: Text prompt for sample generation
        tokenizer: tiktoken tokenizer

    Returns:
        Tuple of (train_losses, val_losses, tokens_seen_list)
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epoch):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (step{global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen
