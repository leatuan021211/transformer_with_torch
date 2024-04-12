import torch.nn as nn

class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()

        self.d_model = d_model # Dimensionality of the model's input and output
        self.d_ff = d_ff # Dimensionality of the inner layer in the feed-forward network

        # Fully connected layers
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        fc = self.fc1(x)
        fc = self.relu(fc)
        fc = self.fc2(fc)
        return fc