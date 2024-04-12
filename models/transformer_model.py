import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from .encoder import EncoderLayer
from .decoder import DecoderLayer
from .posisional_encoding import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, dropout):
        super(TransformerModel, self).__init__()

        self.src_vocab_size = src_vocab_size # Source vocabulary size
        self.tgt_vocab_size = tgt_vocab_size # Target vocabulary size
        self.d_model = d_model # The dimensionality of the model's embeddings
        self.num_heads = num_heads # Number of attention heads in the multi-head attention mechanism
        self.d_ff = d_ff # Dimensionality of the inner layer in the feed-forward network
        self.num_layers = num_layers # Number of layers for both the encoder and the decoder
        self.max_seq_length = max_seq_length # Maximum sequence length for positional encoding
        self.dropout = dropout # Dropout rate for regularization

        # Embedding layer for the source sequence
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        # Embedding layer for the target sequence
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # Positional encoding component
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # A list of encoder layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # A list of decoder layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Final fully connected (linear) layer mapping to target vocabulary size
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def generate_mark(self, src, tgt, device="cpu"):
        """
            This method is used to create masks for the source and target sequences, ensuring that 
            padding tokens are ignored and that future tokens are not visible during training for the target sequence
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(device)
        return src_mask, tgt_mask
    
    def forward(self, src, tgt, device="cpu"):

        # The source and target sequences are first embedded using their respective embedding layers 
        # and then added to their positional encodings
        src_mask, tgt_mask = self.generate_mark(src, tgt, device="cpu")
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # The source sequence is passed through the encoder layers, with the final encoder output 
        # representing the processed source sequence
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # The target sequence and the encoder's output are passed through the decoder layers, resulting in the decoder's output
        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

        # The decoder's output is mapped to the target vocabulary size using a fully connected (linear) layer
        output = self.fc(dec_output)
        return output