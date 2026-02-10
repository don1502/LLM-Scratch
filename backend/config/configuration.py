# Configuration for the model..

class Config:
    "vocab_size" = 50257,
    "context_length" = 1024,
    "emb_dim" = 768,    # Embedding Dimension
    "n_heads" = 12,  # Numbers of attention head
    "n_layers" = 12,    # Number of layers
    "drop_rate" = 0.1,
    "qkv_bias" = False