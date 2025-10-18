from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn.functional as F
import torch
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_dim):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)



    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.bn2(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.bn3(x)

        x= global_mean_pool(x, batch)
        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x #return probabilities for binary classification

if __name__ == "__main__":
    print("This file only contains the GCN model definition.")
    print("Please run the GNN_SignalvsBackground.py script to train and evaluate the model.")
    # model = GCN(input_size = 10, hidden_dim = 64)
    # print("example model initisation: ", + model)
