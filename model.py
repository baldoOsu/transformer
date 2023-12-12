import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_length, d_model)
        # Vector of shape (seq_length, 1)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        # Vector of shape (1, d_model/2)
        # Denominator is 10000^(2i/d_model), following Attention is All You Need page 6
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # All odd indices use sin, all even indices use cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension (1, seq_length, d_model)
        pe = pe.unsqueeze(0)
        
        # Register buffer is a tensor that is not a model parameter
        # Meaning it will not be updated during backpropagation
        # But will still be moved to device when calling model.to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # PE is static and not learnable
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        # Get mean and std, while preserving shape
        # Preserving shape is important for doing math with x
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Layer norm formula, as seen here https://leimao.github.io/blog/Layer-Normalization/
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, seq_length, d_model) --> (Batch, seq_length, d_ff) --> (Batch, seq_length, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

