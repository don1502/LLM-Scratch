# Importing necessary libraries...
from MultiheadAttention import MultiHeadAttention
from feedForward import FeedForward
from layerNormalization import LayerNorm
import torch.nn as nn
from config.configuration import Config


# Implementation of transformer block...

cfg = Config()

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_head=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        # Shortcut Connection for attention block

        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Shortcut connection for FeedForward NN

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        self.drop_shortcut(x)
        x = x + shortcut
        return x