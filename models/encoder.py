import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

from .multihead_attention import MultiHeadAttention
from .position_wise_feedforward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model # The dimensionality of the input
        self.num_heads = num_heads # The number of attention heads in the multi-head attention
        self.d_ff = d_ff # The dimensionality of the inner layer in the position-wise feed-forward network
        self.dropout = dropout # TThe dropout rate used for regularization

        # Multi-head attention mechanism
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Position-wise feed-forward neural network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        # Layer normalization, applied to smooth the layer's input
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layer, used to prevent overfitting by randomly setting some activations to zero during training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: The input to the encoder layer.
            mask: Optional mask to ignore certain parts of the input.

        Processing steps:
            1. Self-Attention: The input x is passed through the multi-head self-attention mechanism.

            2. Add & Normalize (after Attention): The attention output is added to the original input (residual connection), 
            followed by dropout and normalization using norm1.
            
            3. Feed-Forward Network: The output from the previous step is passed through the position-wise feed-forward network.
            
            4. Add & Normalize (after Feed-Forward): Similar to step 2, the feed-forward output is added to 
            the input of this stage (residual connection), followed by dropout and normalization using norm2.
            
            5. Output: The processed tensor is returned as the output of the encoder layer.
        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x