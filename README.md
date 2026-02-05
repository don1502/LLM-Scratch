# LLM From Scratch (PyTorch)

This project is a minimal implementation of a **GPT-style Large Language Model** built from scratch using **PyTorch**.  
It is designed for learning and experimentation, not production use.

---

##  What This Model Does

- Uses **BPE tokenization**
- Implements **Multi head self-attention, feed-forward layers, and layer normalization**
- Trains a **GPT-like transformer** on raw text
- Generates text autoregressively
  

---

Each module is intentionally separated to keep the codebase **clean, modular, and extensible**.

---

##  Tokenization

The model uses a **character-level tokenizer**, which:

- Converts each unique character into an integer ID
- Creates a small, fixed vocabulary
- Makes the learning process easier to understand
- Avoids external dependencies like BPE or SentencePiece

While inefficient for large-scale models, character tokenization is ideal for **learning and experimentation**.

---

##  Model Architecture

The model follows a **GPT-style Transformer architecture**:

1. **Token Embedding**  
   Converts token IDs into dense vectors.

2. **Positional Embedding**  
   Adds information about token positions in the sequence.

3. **Transformer Blocks** (stacked)
   - Masked self-attention
   - Feed-forward network
   - Residual connections
   - Layer normalization

4. **Language Modeling Head**  
   Projects hidden states to vocabulary logits.

The model is trained using **causal (autoregressive) language modeling**, where it predicts the next token given previous tokens.

---

##  Training Objective

The training task is **next-token prediction**:

> Given a sequence of tokens  
> Predict the next token at every position

- Loss function: **Cross-Entropy Loss**
- Optimizer: **AdamW**
- Training data is sampled in fixed-length blocks
- Gradients are backpropagated through time

---

##  Configuration

All important parameters are defined in one place:

Examples:

- Batch size
- Context length (block size)
- Embedding dimension
- Number of attention heads
- Number of transformer layers
- Learning rate
- Training iterations

This makes experimentation simple and reproducible.

---
