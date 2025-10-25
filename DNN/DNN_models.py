"""
Cole Looney 25/10/2025

DNN_models.py

Define Deep Neural Network Model and Early Stopper

Arch:
Three layers with 64 hidden dimensions, binary classification
"""


import torch.nn as nn
hidden_dim = 64
output_dim = 1

class DNN(nn.Module):
    """
    define model architecture
    """
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
    
class EarlyStopper:
    """
    Class to stop training early

    patience: how many times validation loss can increase before early stopping kicks in
    min_delta = ignore small increases in val loss
    """
    def __init__(self,patience = 1, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False