import h5py
import numpy as np
import pandas as pd
import torch
import os.path as osp
from torch_geometric.data import Dataset, Data
from sklearn.neighbors import kneighbors_graph

hfivesdir = '../data/new_Input_CP_Studies_llqq_LinearTerm_13th_October2025.h5'
graphsdir = "../graphdata/CP_Studies_llqq_graphs"

class CPDataSet(Dataset):
    def __init__(self,root, transform = None, pre_transform = None, pre_filter = None):
        self.event_data = None #store as attribute to access later
        super().__init__(root,transform,pre_transform,pre_filter) #calls process if data not processed

        if osp.exists(self.processed_paths[1]): #load event data info if it exists
            self.event_data = torch.load(self.processed_paths[1], weights_only = False)

    
    @property
    def raw_file_names(self):
        return [osp.basename(hfivesdir)]
    
    @property
    def processed_file_names(self):
        return ['data_0.pt', 'event_data_info.pt']
    
    def download(self):
        pass


    def process(self):
        # load raw data
        with h5py.File(hfivesdir, 'r') as f:
            df_1d = pd.DataFrame(f['LargeRJet']['1d'][:])
            df_1d['lumi_label'] = 0 #initialize new column for labels
            df_1d.loc[df_1d['Lumi_weight'] > 0, 'lumi_label'] = 1 #label 1 for signal
            event_data_local = df_1d

            raw_constituents = f['LargeRJet']['2d'][:] #unstructued array of constituents

        torch.save(event_data_local, self.processed_paths[1]) #save event data info for later use in label

        constant_features = np.zeros(raw_constituents.shape + (len(raw_constituents.dtype.names),), dtype=np.float32)
        feature_names = raw_constituents.dtype.names

        for i,name in enumerate(feature_names):
            constant_features[...,i] = raw_constituents[name]

        valid_mask = constant_features[:,:,-1] > 0 #mask for valid constituents based on pT > 0
        constant_features = np.nan_to_num(constant_features, nan=0.0) #fill NaNs with 0

        num_events = constant_features.shape[0] #number fo events
        for i in range(num_events):
            mask = valid_mask[i] #mask for valid constituents in event i
            valid_nodes = constant_features[i,mask,:] #get valid constituents for event i

            lumi_weight_val = event_data_local['Lumi_weight'].iloc[i]
            lumi_weight_tensor = torch.tensor([lumi_weight_val], dtype=torch.float)

            if valid_nodes.shape[0] < 2:
                continue
            

            node_feats = self.__get_node_features(valid_nodes) #get node features tensor
            edge_index = self.__get_edge_index(valid_nodes,feature_names) #get edge index tensor

            label = self._get_labels(i, event_data_local) #get label tensor

            data = Data(x = node_feats, edge_index = edge_index, y = label, lumi_weight = lumi_weight_tensor) #create PyG Data object

            torch.save(data, osp.join(self.processed_dir, f'data_{i}.pt')) #save graph data object

            if (i + 1) % 5000 == 0:
                print(f'Processed {i+1}/{num_events} events.')
    

    def __get_node_features(self, valid_nodes):
        return torch.tensor(valid_nodes, dtype = torch.float) #all features for now
    
    def __get_edge_index(self, valid_nodes, feature_names, k_neighbors = 6):
        # use eta and phi to construct k-NN graph
        eta_idx = feature_names.index('constituent_eta')
        phi_idx = feature_names.index('constituent_phi')

        eta_phi_coords = valid_nodes[:, [eta_idx, phi_idx]] #extract eta and phi

        num_constituents = valid_nodes.shape[0] #number of valid constituents

        if num_constituents < 2:
            return torch.tensor([], dtype=torch.long).reshape(2, 0)
            
        actual_k = min(k_neighbors, num_constituents  - 1) #adjust k if fewer constituents

        #create k-NN graph using sklearn
        adjacency_matrix = kneighbors_graph(
            eta_phi_coords, 
            n_neighbors=actual_k, 
            mode='connectivity', 
            include_self=False
        )

        # convert to edge index format
        edge_index_sparse = adjacency_matrix.tocoo()
        return torch.tensor(
            np.array([edge_index_sparse.row, edge_index_sparse.col]), 
            dtype=torch.long
        )

    def _get_labels(self, i,event_data_df): #get label for event i
        label_val = event_data_df['lumi_label'].iloc[i]
        return torch.tensor([label_val], dtype=torch.long)

    def len(self):
        return len(self.event_data) if self.event_data is not None else 0
    
    def get(self, idx): #get graph data object for event idx
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only = False)
        return data
    
if __name__ == '__main__':
    dataset = CPDataSet(root = graphsdir) #initialize and process dataset if needed

    print(f"\nDataset loaded successfully!")
    print(f"Number of graphs: {len(dataset)}") #number of events with valid graphs
    print(f"Example graph:\n{dataset[0]}")
    print(f"Node features shape: {dataset[0].x.shape}")
    print(f"Edge index shape: {dataset[0].edge_index.shape}")
