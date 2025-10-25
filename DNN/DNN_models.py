
import torch.nn as nn

from preprocess_large_data import X_train_tensor
hidden_dim = 64
output_dim = 1
input_size = X_train_tensor.shape[0]

class DNN(nn.Module):
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