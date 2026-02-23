"""
Fine-tune a pre-trained GPT-2 model on scraped Wikipedia technology data.

This script:
1. Loads pre-trained GPT-2 124M weights
2. Loads the scraped technology corpus
3. Fine-tunes the model with lower learning rate
4. Generates sample text to show improvement
5. Saves the fine-tuned model

Usage:
    python finetune.py

Note: Requires GPT-2 weights to be downloaded first.
      Run the generate.py load_model() function or
      use gpt_download3.py to download weights.
"""

import os
import sys
import time
import torch
import tiktoken

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from GPTmodel import GPTModel, GPT_Config_124m
from data.dataset import create_dataloader
from training.trainer import (
    train_model_simple,
    text_to_token,
    token_to_text,
    generate_text,
)
from scraper.wikipedia_scraper import WikipediaScraper
from scraper.data_processor import prepare_training_data


def load_pretrained_model():
    """Load pre-trained GPT-2 weights using the existing infrastructure."""
    from generate import load_model

    print("Loading pre-trained GPT-2 124M model...")
    model, tokenizer = load_model(models_dir="gpt_models", model_size="124M")
    return model, tokenizer


def generate_sample(model, tokenizer, prompt, device, max_tokens=100):
    """Generate a sample text from a prompt."""
    model.eval()
    model.to(device)
    input_ids = text_to_token(prompt, tokenizer).to(device)

    with torch.no_grad():
        output_ids = generate_text(
            model=model,
            idx=input_ids,
            max_new_token=max_tokens,
            context_size=1024,
        )
    return token_to_text(output_ids, tokenizer)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Step 1: Scrape data if needed ----
    corpus_path = os.path.join("scraped_data", "technology_corpus.txt")
    cleaned_path = os.path.join("scraped_data", "technology_cleaned.txt")

    if not os.path.exists(corpus_path):
        print("\n" + "=" * 60)
        print("Step 1: Scraping Wikipedia articles...")
        print("=" * 60)
        scraper = WikipediaScraper()
        scraper.scrape(max_articles=30, links_per_topic=2, delay=1.0)
    else:
        print(f"Corpus already exists at {corpus_path}")

    # ---- Step 2: Load and prepare data ----
    print("\n" + "=" * 60)
    print("Step 2: Preparing training data...")
    print("=" * 60)
    data = prepare_training_data(corpus_path, cleaned_path)

    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(data))
    print(f"Total tokens for fine-tuning: {total_tokens:,}")

    # ---- Step 3: Load pre-trained model ----
    print("\n" + "=" * 60)
    print("Step 3: Loading pre-trained GPT-2 model...")
    print("=" * 60)

    try:
        model, tokenizer = load_pretrained_model()
    except Exception as e:
        print(f"\nError loading pre-trained model: {e}")
        print("\nTo download GPT-2 weights, please run:")
        print("  python gpt_download3.py")
        print("\nOr use the download_and_load_gpt2 function from gpt_download3.py")
        return

    model.to(device)

    # ---- Step 4: Show pre-fine-tuning samples ----
    print("\n" + "=" * 60)
    print("BEFORE fine-tuning - sample generations:")
    print("=" * 60)

    test_prompts = [
        "Artificial intelligence is",
        "Machine learning algorithms can",
        "The future of quantum computing",
    ]

    for prompt in test_prompts:
        output = generate_sample(model, tokenizer, prompt, device, max_tokens=50)
        print(f"\nPrompt: '{prompt}'")
        print(f"Output: {output}")

    # ---- Step 5: Create dataloaders ----
    print("\n" + "=" * 60)
    print("Step 5: Creating dataloaders for fine-tuning...")
    print("=" * 60)

    train_ratio = 0.85
    split_idx = int(train_ratio * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Use smaller context for fine-tuning to fit in memory
    ft_context_len = 256
    train_loader = create_dataloader(
        train_data,
        tokenizer,
        batch_size=2,
        max_len=ft_context_len,
        stride=ft_context_len,
        shuffle=True,
        drop_last=True,
    )

    val_loader = create_dataloader(
        val_data,
        tokenizer,
        batch_size=2,
        max_len=ft_context_len,
        stride=ft_context_len,
        shuffle=False,
        drop_last=False,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    if len(train_loader) == 0:
        print("ERROR: Not enough data for fine-tuning.")
        return

    # ---- Step 6: Fine-tune ----
    print("\n" + "=" * 60)
    print("Step 6: Fine-tuning GPT-2 on technology data...")
    print("=" * 60)

    # Lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )

    num_epochs = 5
    start_time = time.time()

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epoch=num_epochs,
        eval_freq=10,
        eval_iter=5,
        start_context="Artificial intelligence is",
        tokenizer=tokenizer,
    )

    elapsed = (time.time() - start_time) / 60
    print(f"\nFine-tuning completed in {elapsed:.2f} minutes")

    # ---- Step 7: Show post-fine-tuning samples ----
    print("\n" + "=" * 60)
    print("AFTER fine-tuning - sample generations:")
    print("=" * 60)

    for prompt in test_prompts:
        output = generate_sample(model, tokenizer, prompt, device, max_tokens=50)
        print(f"\nPrompt: '{prompt}'")
        print(f"Output: {output}")

    # ---- Step 8: Save fine-tuned model ----
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", "gpt2_finetuned_technology.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": GPT_Config_124m,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        checkpoint_path,
    )
    print(f"\nFine-tuned model saved to: {checkpoint_path}")
    print("Done!")


if __name__ == "__main__":
    main()
