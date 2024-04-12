import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model # The dimension of the model's input
        self.max_seq_length = max_seq_length # The maximum length of the sequence for which positional encodings are pre-computed

        pe = torch.zeros(max_seq_length, d_model) # A tensor filled with zeros, which will be populated with positional encodings
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # A tensor containing the position indices for each position in the sequence
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # A term used to scale the position indices in a specific way

        # The sine function is applied to the even indices and the cosine function to the odd indices of pe
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Finally, pe is registered as a buffer, which means it will be part of 
        # the module's state but will not be considered a trainable parameter
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
            The forward method simply adds the positional encodings to the input x.

        It uses the first x.size(1) elements of pe to ensure that the positional encodings match the actual sequence length of x
        """
        return x + self.pe[:, :x.size(1)]