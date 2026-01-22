# Importing the necessary libraries...
import torch.nn as nn
from geluFunction import GELU

# Implementation of Feedforward neural network

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Sequential(  # Structure of feedforward neural network is linear -> GeLU -> Linear
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]), # Using GPT configuration for FeedForward NN  This line is Expansion layer of feedforward NN
            GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]), # This line is contraction layer of FeedForward NN
        )

    def forward(self, x):
        return self.layer(x)