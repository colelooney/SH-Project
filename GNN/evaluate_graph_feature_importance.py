from GNN import GCN
import torch
from preprocess_graph_data import CPDataSet
from torch_geometric.data import DataLoader
import pandas as pd
import os.path as osp
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from sklearn.feature_selection import mutual_info_classif

model_path = 'ModelsGNN/gnn_model.pth'
dataset =  CPDataSet(root = '../graphdata/CP_Studies_llqq_graphs')

torch.manual_seed(12345)
dataset = dataset.shuffle()
input_size = dataset.num_node_features

model = GCN(input_size, hidden_dim = 64)

batch_size = 256
test_dataset = dataset[int(0.25 * len(dataset)):]

test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

model.load_state_dict(torch.load(model_path))
torch.no_grad()

model.eval()



