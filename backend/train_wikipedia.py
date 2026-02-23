"""
Train a GPT model on scraped Wikipedia technology data.

This script:
1. Scrapes Wikipedia articles on technology topics (if not already done)
2. Prepares the data for training
3. Trains a GPT model from scratch
4. Saves the trained model checkpoint

Usage:
    python train_wikipedia.py
"""

import os
import sys
import time
import torch
import tiktoken

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from GPTmodel import GPTModel
from data.dataset import create_dataloader
from training.trainer import train_model_simple
from scraper.wikipedia_scraper import WikipediaScraper
from scraper.data_processor import prepare_training_data


# Smaller config for training from scratch on limited data
GPT_Config_Small = {
    "vocab_size": 50257,
    "context_length": 256,   # Smaller context for faster training
    "emb_dim": 256,          # Smaller embedding for limited data
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def main():
    # ---- Step 1: Scrape data if not available ----
    corpus_path = os.path.join("scraped_data", "technology_corpus.txt")
    cleaned_path = os.path.join("scraped_data", "technology_cleaned.txt")

    if not os.path.exists(corpus_path):
        print("=" * 60)
        print("Step 1: Scraping Wikipedia articles...")
        print("=" * 60)
        scraper = WikipediaScraper()
        scraper.scrape(max_articles=30, links_per_topic=2, delay=1.0)
    else:
        print(f"Corpus already exists at {corpus_path}")

    # ---- Step 2: Prepare data ----
    print("\n" + "=" * 60)
    print("Step 2: Preparing training data...")
    print("=" * 60)
    data = prepare_training_data(corpus_path, cleaned_path)

    if len(data) < 1000:
        print("WARNING: Very little data scraped. Training may not be effective.")
        print("Consider running the scraper with more articles.")

    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(data))
    print(f"Total tokens: {total_tokens:,}")

    # ---- Step 3: Create dataloaders ----
    print("\n" + "=" * 60)
    print("Step 3: Creating dataloaders...")
    print("=" * 60)

    train_ratio = 0.85
    split_idx = int(train_ratio * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    cfg = GPT_Config_Small
    train_loader = create_dataloader(
        train_data,
        tokenizer,
        batch_size=4,
        max_len=cfg["context_length"],
        stride=cfg["context_length"],
        shuffle=True,
        drop_last=True,
    )

    val_loader = create_dataloader(
        val_data,
        tokenizer,
        batch_size=4,
        max_len=cfg["context_length"],
        stride=cfg["context_length"],
        shuffle=False,
        drop_last=False,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    if len(train_loader) == 0:
        print("ERROR: Not enough data for training. Need more scraped text.")
        return

    # ---- Step 4: Train model ----
    print("\n" + "=" * 60)
    print("Step 4: Training GPT model...")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(123)
    model = GPTModel(cfg)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0004, weight_decay=0.1
    )

    num_epochs = 15
    start_time = time.time()

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epoch=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Artificial intelligence is",
        tokenizer=tokenizer,
    )

    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining completed in {elapsed:.2f} minutes")

    # ---- Step 5: Save checkpoint ----
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", "gpt_technology.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        checkpoint_path,
    )
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    print("Done!")


if __name__ == "__main__":
    main()
