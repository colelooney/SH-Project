from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv, global_add_pool, global_max_pool
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch_cluster import knn_graph

class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_dim,dropout = 0.2):
        super().__init__()
        torch.manual_seed(12345)

        p = 0.2 # dropout probability
        self.conv1 = GCNConv(input_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_dim, hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)

        self.conv3 = GCNConv(hidden_dim//2, hidden_dim//4)
        self.bn3 = nn.BatchNorm1d(hidden_dim//4)

        self.classifier = nn.Linear(hidden_dim//4, 1)
        self.dropout = nn.Dropout(p)



    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.bn2(x)
        # x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.bn3(x)
        # x = self.relu(x)
        x = self.dropout(x)

        x= global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.classifier(x)
        # x = torch.sigmoid(x)

        return x #return probabilities for binary classification

# class GCN(nn.Module):
#     def __init__(self, input_size, hidden_dim=256, dropout=0.2):
#         super().__init__()
#         torch.manual_seed(12345)

#         self.conv1 = GCNConv(input_size, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)

#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)

#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.bn3 = nn.BatchNorm1d(hidden_dim)

#         # Residual projection for skip connections
#         self.res_fc = nn.Linear(input_size, hidden_dim)

#         self.dropout = nn.Dropout(dropout)
#         self.act = nn.LeakyReLU(0.1)
#         self.identity = nn.Identity()

#         # Post-pooling MLP
#         self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim // 2)
#         self.bn_fc1 = nn.BatchNorm1d(hidden_dim // 2)
#         self.fc2 = nn.Linear(hidden_dim // 2, 1)

#     def forward(self, x, edge_index, batch):
#         # Residual connection
#         res = self.res_fc(x)

#         # 1st conv
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = self.act(x)
#         # x = self.dropout(x)
#         x = self.identity(x)
#         x = x + res  # add skip

#         # 2nd conv
#         res = x
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = self.act(x)
#         # x = self.dropout(x)
#         x = self.identity(x)
#         x = x + res

#         # 3rd conv
#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = self.act(x)
#         # x = self.dropout(x)
#         x = self.identity(x)

#         # Global pooling: mean + max
#         x_mean = global_add_pool(x, batch)
#         x_max = global_max_pool(x, batch)
#         x = torch.cat([x_mean, x_max], dim=1)

#         # MLP head
#         x = self.fc1(x)
#         x = self.bn_fc1(x)
#         x = self.act(x)
#         # x = self.dropout(x)
#         x = self.identity(x)
#         x = self.fc2(x)

#         return x

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

if __name__ == "__main__":
    print("This file only contains the GCN model definition.")
    print("Please run the GNN_SignalvsBackground.py script to train and evaluate the model.")
    # model = GCN(input_size = 10, hidden_dim = 64)
    # print("example model initisation: ", + model)
