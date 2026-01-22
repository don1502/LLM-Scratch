# Importing the necessary libraries...
import torch
import torch.nn as nn


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