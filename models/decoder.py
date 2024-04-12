import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

from .position_wise_feedforward import PositionWiseFeedForward
from .multihead_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model # The dimensionality of the input
        self.num_heads = num_heads # The number of attention heads in the multi-head attention
        self.d_ff = d_ff # The dimensionality of the inner layer in the feed-forward network
        self.dropout = dropout # Dropout rate for regularization

        # Multi-head self-attention mechanism for the target sequence
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Multi-head attention mechanism that attends to the encoder's output
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        # Position-wise feed-forward neural network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        # Layer normalization components
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_ouput, src_mark, tgt_mark):
        """
            Args:
                x: The input to the decoder layer.
                enc_output: The output from the corresponding encoder (used in the cross-attention step).
                src_mask: Source mask to ignore certain parts of the encoder's output.
                tgt_mask: Target mask to ignore certain parts of the decoder's input.
            
        """
        # The input x is processed through a self-attention mechanism
        attn_output = self.self_attn(x, x, x, tgt_mark)

        # The output from self-attention is added to the original x, followed by dropout and normalization using norm1
        # The normalized output from the previous step is processed through a cross-attention mechanism 
        # that attends to the encoder's output enc_output
        x = self.norm1(x + self.dropout(attn_output))

        # The output from cross-attention is added to the input of this stage, followed by dropout and normalization using norm2
        attn_output = self.cross_attn(x, enc_ouput, enc_ouput, src_mark)

        # The output from the previous step is passed through the feed-forward network
        x = self.norm2(x + self.dropout(attn_output))

        # The feed-forward output is added to the input of this stage, followed by dropout and normalization using norm3
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x