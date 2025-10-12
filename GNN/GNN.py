from torch_geometric.nn import GCNConv
from torch.nn.functional import F
import torch

class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_dim):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_size, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = F.dropout(x, p =0.5,training = self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x) #return probabilities for binary classification

if __name__ == "__main__":
    print("This file only contains the GCN model definition.")
    print("Please run the GNN_SignalvsBackground.py script to train and evaluate the model.")
    model = GCN(input_size = 10, hidden_dim = 64)
    print("example model initisation: ", + model)
