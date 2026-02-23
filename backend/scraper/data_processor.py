"""
Data processor for cleaning and preparing scraped Wikipedia text
for GPT training.
"""

import re
import os


def load_corpus(filepath):
    """Load the scraped corpus from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def clean_corpus(text):
    """
    Additional cleaning of the scraped corpus for training.

    Args:
        text: Raw corpus text

    Returns:
        Cleaned text suitable for GPT training
    """
    # Remove markdown headers
    text = re.sub(r"^# .+$", "", text, flags=re.MULTILINE)

    # Remove separator lines
    text = text.replace("---", "")

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove very short paragraphs
    paragraphs = text.split("\n\n")
    filtered = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    text = "\n\n".join(filtered)

    return text.strip()


def prepare_training_data(input_path, output_path=None):
    """
    Load, clean, and save training data.

    Args:
        input_path: Path to scraped corpus
        output_path: Path to save cleaned data (optional)

    Returns:
        Cleaned text string
    """
    text = load_corpus(input_path)
    cleaned = clean_corpus(text)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Total characters: {len(cleaned):,}")

    return cleaned


if __name__ == "__main__":
    prepare_training_data(
        "scraped_data/technology_corpus.txt",
        "scraped_data/technology_cleaned.txt",
    )
