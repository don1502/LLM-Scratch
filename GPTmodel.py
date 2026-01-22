from transformer.transformerBlock import TransformerBlock
from transformer.layerNormalization import LayerNorm
import torch
import torch.nn as nn
import tiktoken

# Implementation of 124M version of GPT-2

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    
    def forward(self, in_index):
        batch_size, seq_len = in_index.shape
        tok_embeds = self.tok_emb(in_index)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_index.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x) 
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

GPT_Config_124m = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,    # Embedding Dimension
    "n_heads" : 12,  # Numbers of attention head
    "n_layers" : 12,    # Number of layers
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1  = "Every efforts moves you"
text2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch = torch.stack(batch, dim=0)
print("batch size : \n", batch, "\n")

torch.manual_seed(123)
gptmodel = GPTModel(GPT_Config_124m)
out = gptmodel(batch)
print("Output shape : \n", out.shape,"\n")
print("Logits : \n", out, "\n")