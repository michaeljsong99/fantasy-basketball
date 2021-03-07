"""
A module to create an Artificial Neural Network using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNeuralNetModel, self).__init__()

        self.config = Config()

        self.input_size = input_dim
        self.output_size = 1

        # Create the hidden layers from the config file.
        self.hidden_layers = nn.ModuleList()
        layer_size = input_dim
        for hidden_layer_size in self.config.hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(layer_size, hidden_layer_size))
            layer_size = hidden_layer_size

        self.output_layer = nn.Linear(layer_size, self.output_size)

        # Activation function
        self.activation = self.config.activation

        self.lr_slope = 0.01

        # Dropout rate
        if self.config.dropout:
            self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        for layer in self.hidden_layers:
            # Feed the data through a linear layer and activation layer.
            if self.activation == "relu":
                x = F.relu_(layer(x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu_(layer(x), negative_slope=self.lr_slope)
            elif self.activation == "tanh":
                x = F.tanh(layer(x))
            elif self.activation == "sigmoid":
                x = F.sigmoid(layer(x))
            else:
                raise RuntimeError(f"Invalid activation function: {self.activation}")

        if self.config.dropout:
            x = self.dropout(x)

        return self.output_layer(x)
