"""
Cole Looney 25/10/2025

DNN_models.py

Define Deep Neural Network Model and Early Stopper

Arch:
Four layers with 128 hidden dimensions, binary classification
"""


import torch.nn as nn
hidden_dim = 512
output_dim = 1
dropout_rate = 0.2

class DNN(nn.Module):
    """
    define model architecture
    """
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Identity(),
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Identity(),
            nn.Linear(hidden_dim//2,hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Identity(),
            nn.Linear(hidden_dim//4,hidden_dim//8),
            nn.BatchNorm1d(hidden_dim//8),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Identity(),
            nn.Linear(hidden_dim//8,output_dim)
        )

    def forward(self, x):
        return self.layers(x)
    
class EarlyStopper:
    """
    Class to stop training early

    patience: how many times validation loss can increase before early stopping kicks in
    min_delta = ignore small increases in val loss
    """
    def __init__(self,patience = 5, min_delta = 1e-4):
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

class EarlyStopperAUC:
    """
    Class to stop training early based on ROC_AUC score

    patience: how many times ROC AUC can decrease before early stopping kicks in
    min_delta = ignore small decrease in roc
    """
    def __init__(self,patience = 5, min_delta = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_auc = -float('inf')

    def early_stop(self, auc):
        if auc > self.best_auc + self.min_delta:
            self.best_auc = auc
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience