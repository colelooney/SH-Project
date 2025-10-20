from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch_cluster import knn_graph

class EdgeConvBlock(nn.Module):
    def __init__(self, in_size,layer_size):
        super(EdgeConvBlock, self).__init__()
        
        layers = []
        layers.append(nn.Linear(in_size * 2, layer_size))
        layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for i in range(2):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.model)

class ParticleNet(nn.Module):
    def __init__(self, kernel_sizes, fc_size, dropout, k, node_feat_size, num_classes=2):
        super(ParticleNet, self).__init__()
        self.node_feat_size = node_feat_size
        self.num_classes = num_classes

        self.k = k
        self.num_edge_convs = len(kernel_sizes)
        self.kernel_sizes = kernel_sizes
        self.fc_size = fc_size
        self.dropout = dropout

        self.edge_nets = nn.ModuleList()
        self.edge_convs = nn.ModuleList()

        self.kernel_sizes.insert(0, self.node_feat_size)
        self.output_sizes = np.cumsum(self.kernel_sizes)

        self.edge_nets.append(EdgeConvBlock(self.node_feat_size, self.kernel_sizes[1]))
        self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr="mean"))

        for i in range(1, self.num_edge_convs):
            self.edge_nets.append(EdgeConvBlock(self.output_sizes[i], self.kernel_sizes[i+1]))
            self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr="mean"))
        
        self.fc1 = nn.Sequential(nn.Linear(self.output_sizes[-1], 
                                self.fc_size))
        self.dropout_layer = nn.Dropout(p = self.dropout)
        self.fc2 = nn.Linear(self.fc_size, self.num_classes)

    def forward(self, data):
        x = data.x
        batch = data.batch
        
        #extracting delta phi and delta eta for initial node position 
        pos = data.x[:,[1,0]] 

        for i in range(self.num_edge_convs):
            edge_index = (knn_graph(pos, self.k, batch) if i==0 else knn_graph(x, self.k, batch)) #graphs with knn 

            x = torch.cat((self.edge_convs[i](x, edge_index), x), dim=1)
        
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.dropout_layer(x)

        return self.fc2(x)


# class GCN(torch.nn.Module):
#     def __init__(self, input_size, hidden_dim):
#         super().__init__()
#         torch.manual_seed(12345)

#         p = 0.2 # dropout probability
#         self.conv1 = GCNConv(input_size, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.relu = nn.ReLU()
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)

#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.bn3 = nn.BatchNorm1d(hidden_dim)

#         self.conv4 = GCNConv(hidden_dim, hidden_dim)
#         self.bn4 = nn.BatchNorm1d(hidden_dim)
#         self.classifier = nn.Linear(hidden_dim, 1)
#         self.dropout = nn.Dropout(p)



#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = self.relu(x)
#         x = self.bn1(x)
#         x = self.dropout(x)

#         x = self.conv2(x, edge_index)
#         x = self.relu(x)
#         x = self.bn2(x)
#         x = self.dropout(x)

#         x = self.conv3(x, edge_index)
#         x = self.relu(x)
#         x = self.bn3(x)
#         x = self.dropout(x)

#         x= global_mean_pool(x, batch)
#         x = self.dropout(x)
#         x = self.classifier(x)
#         x = torch.sigmoid(x)

#         return x #return probabilities for binary classification

# if __name__ == "__main__":
#     print("This file only contains the GCN model definition.")
#     print("Please run the GNN_SignalvsBackground.py script to train and evaluate the model.")
#     # model = GCN(input_size = 10, hidden_dim = 64)
#     # print("example model initisation: ", + model)
