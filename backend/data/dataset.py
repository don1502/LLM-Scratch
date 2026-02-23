"""
Dataset and DataLoader utilities for GPT pre-training.
Implements sliding-window tokenized datasets from raw text.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """
    A dataset that creates input-target pairs for GPT training
    using a sliding window approach over tokenized text.
    """

    def __init__(self, txt, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        # Using sliding window to chunk block into overlapping sequence of max_len
        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i : i + max_len]
            target_chunk = token_ids[i + 1 : i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt,
    tokenizer,
    batch_size=4,
    max_len=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """
    Create a DataLoader for GPT training from raw text.

    Args:
        txt: Raw text string
        tokenizer: tiktoken tokenizer
        batch_size: Batch size
        max_len: Maximum sequence length (context window)
        stride: Stride for sliding window
        shuffle: Whether to shuffle data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of data loading workers

    Returns:
        DataLoader instance
    """
    dataset = GPTDatasetV1(txt, tokenizer, max_len, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
