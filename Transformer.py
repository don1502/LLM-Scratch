# Complete Transformer Block Implementation

import torch
import torch.nn as nn

GPT_Config_124m = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,    # Embedding Dimension
    "n_heads" : 12,  # Numbers of attention head
    "n_layers" : 12,    # Number of layers
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased = False)
        norm_x = (x-mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))* (x + 0.044715 * torch.pow(x, 3))))
 
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
    
# Implementation of Multi head attention mechanism with weight split

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_head, qkv_bias = False):
        super().__init__()
        assert(d_out % num_head == 0),\
            "d_out must be divisible by num_head"
        self.d_out = d_out
        self.num_head = num_head
        self.head_dim = d_out//num_head # Reduce the production dim to match the desire output dim
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_in,d_out) # linear projection
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length),diagonal=1))
        
    def forward(self, x):
        b, num_token, d_in = x.shape
        key =  self.w_key(x)    
        query = self.w_query(x)
        value = self.w_value(x) 

        # We implicity split the matrix by adding a num_head dimension 
        # unroll last dim : (b, num_token, d_out) -> (b, num_token, num_head, head_dim)
        key = key.view(b, num_token, self.num_head, self.head_dim)
        value = value.view(b, num_token, self.num_head, self.head_dim)
        query = query.view(b, num_token, self.num_head, self.head_dim)

        # Transpose: (b, num_token, num_head, head_dim) -> (b, num_head, num_token, head_dim)
        key = key.transpose(1,2)
        query = query.transpose(1,2)
        value = value.transpose(1,2)

        # Compute scaled dot product attention aka self attention with causal mask
        attention_score = query @ key.transpose(2,3)

        # original mask truncated to the number of tokens and convert to boolean 
        mask_bool = self.mask.bool()[:num_token, :num_token]

        attention_score.masked_fill(mask_bool, -torch.inf) # masking the attention score

        attention_weight = torch.softmax(attention_score/key.shape[-1]**0.5, dim = -1)
        attention_weight = self.dropout(attention_weight)

        context_vector_multihead = (attention_weight @ value).transpose(1,2)

        # Combine head, where self.d_out = self.num_head * self.head_dim
        context_vector_multihead = context_vector_multihead.contiguous().view(b, num_token, self.d_out)
        context_vector_multihead = self.out_proj(context_vector_multihead) # Optional projection

        return context_vector_multihead

# Now let us implement the entire transformer block.....

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

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_Config_124m)
output = block(x)
print("Output Shape : \n", output.shape,"\n")
print("Output : \n", output, "\n")