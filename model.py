import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
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

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h

        # Necessary for splitting into heads
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Lnear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    # This method contains everything from Attention is All You Need, page 5, Scaled Dot Product Attention
    @staticmethod
    def attention(q, k, v, d_k, mask, dropout: nn.Dropout):
        d_k = q.size(-1)

        # 1st MatMul and Scale
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Mask (opt.)
        if mask is not None:
            # Replace all masked values with -1e9 (-inf)
            # Mask decides which words that a word has to pay attention to
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # SoftMax
        # (Batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim = -1)

        # Not on the paper, but dropout is useful for training
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Final MatMul with V
        # Attention scores are returned, because they are useful for visualizing the attention
        return (attention_scores @ value), attention_scores

    # This method contains everything from Attention is All You Need, page 5, Multi-Head Attention
    def forward(self, q, k, v, mask=None):
        # 1st Linear
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split into h heads (Batch, seq_length, d_model) --> (Batch, seq_length, h, d_k) --> (Batch, h, seq_length, d_k)
        query = query.view(query.size(0), query.size(0), self.h, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(0), self.h, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(0), self.h, self.d_k).transpose(1, 2)

        # Scaled Dot Product Attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, self.d_k, mask, self.dropout)

        # Concat
        # (Batch, h, seq_length, d_k) --> (Batch, seq_length, h, d_k) --> (Batch, seq_length, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.h * self.d_k)

        # Final Linear
        # (Batch, seq_length, d_model)
        return self.w_0(x)

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization()

    # Sublayer is a multi-head attention or a ff layer that takes x as input
    def forward(self, x, sublayer):
        # In Attention Is All You Need, layer norm is applied after the sublayer
        # However, more recent studies have shown that pre-normalization works better
        return x + self.dropout(sublayer(self.norm(x)))

# Everything after the positional embedding in the encoder
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super.__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Everything after positional encoding in the decoder block
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_length, d_model) --> (Batch, seq_length, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embeds: InputEmbeddings, tgt_embeds: InputEmbeddings, src_pos: PositionalEmbedding, tgt_pos: PositionalEmbedding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeds = src_embeds
        self.tgt_embeds = tgt_embeds
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, mask):
        src = self.embed(src)
        src = self.pos(src)
        return self.encoder(src, mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.embed(tgt)
        tgt = self.pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1, N: int = 6) -> Transformer:
    # Create embedding layers
    src_embeds = InputEmbeddings(d_model, src_vocab_size)
    tgt_embeds = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional embedding layers
    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout))

    # Create decoder blocks
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout))

    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embeds, tgt_embeds, src_pos, tgt_pos, projection_layer)

    # Finally, initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

