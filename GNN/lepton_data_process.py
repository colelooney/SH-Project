"""
Processes 2d data into PyG Data objects using only leptonic constituents. 
Includes a global edge index rather than KNN approach from before

"""


import h5py
import numpy as np
import pandas as pd
import torch
import os.path as osp
from torch_geometric.data import Dataset, Data

hfivesdir = '../data/s2286706/new_Input_CP_Studies_llqq_LinearTerm_20th_October2025.h5'
graphsdir = "../graphdata/CP_Studies_llqq_graphs_20th_October_Linear"

class CPDataSet(Dataset):
    def __init__(self,root, transform = None, pre_transform = None, pre_filter = None):
        self.event_data = None #store as attribute to access later
        self.feature_names = None
        super().__init__(root,transform,pre_transform,pre_filter) #calls process if data not processed

        if osp.exists(self.processed_paths[1]): #load event data info if it exists
            self.event_data = torch.load(self.processed_paths[1], weights_only = False)
        if osp.exists(self.processed_paths[2]):
            self.feature_names = torch.load(self.processed_paths[2], weights_only = False)

    
    @property
    def raw_file_names(self):
        return [osp.basename(hfivesdir)]
    
    @property
    def processed_file_names(self):
        return ['data_0.pt', 'event_data_info.pt', 'feature_names.pt']
    
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

        num_features = len(raw_constituents.dtype.names)
        constant_features = raw_constituents.view(np.float32).reshape(
            raw_constituents.shape[0], raw_constituents.shape[1], num_features
        )

        feature_names = list(raw_constituents.dtype.names)

        torch.save(feature_names,self.processed_paths[2])

        valid_mask = constant_features[:,:,feature_names.index('constituent_pt')] > 0 #mask for valid constituents based on pT > 0
        constant_features = np.nan_to_num(constant_features, nan=0.0) #fill NaNs with 0

        isLep_idx = feature_names.index('constituent_isLep')

        num_events = constant_features.shape[0] #number fo events
        for i in range(num_events):
            mask = valid_mask[i] #mask for valid constituents in event i
            valid_nodes = constant_features[i,mask,:] #get valid constituents for event i
            lepton_mask_for_event = valid_nodes[:, isLep_idx] == 1
            lepton_nodes = valid_nodes[lepton_mask_for_event]

            if lepton_nodes.shape[0] < 2: #skip if not enough nodes to create an edge
                continue


            node_feats = self.__get_node_features(lepton_nodes) #get node features tensor
            edge_index = self.__get_edge_index(valid_nodes=lepton_nodes.shape[0]) #get edge index tensor
            label = self._get_labels(i, event_data_local) #get label tensor
            lumi_weight_tensor = torch.tensor([event_data_local['Lumi_weight'].iloc[i]],dtype = torch.float)

            data = Data(x = node_feats, edge_index = edge_index, y = label, lumi_weight = lumi_weight_tensor) #create PyG Data object

            torch.save(data, osp.join(self.processed_dir, f'data_{i}.pt')) #save graph data object

            if (i + 1) % 5000 == 0:
                print(f'Processed {i+1}/{num_events} events.')
    

    def __get_node_features(self, valid_nodes):
        return torch.tensor(valid_nodes, dtype = torch.float) #all features for now
    
    def __get_edge_index(self, valid_nodes):
        # connect all nodes => fully connected graph
        i = torch.arange(valid_nodes, dtype=torch.long)
        j = torch.arange(valid_nodes, dtype=torch.long)

        i,j = torch.cartesian_prod(i,j).t()

        mask = i !=j
        edge_index = torch.stack([i[mask],j[mask]],dim=0)
        return edge_index

    def _get_labels(self, i,event_data_df): #get label for event i
        label_val = event_data_df['lumi_label'].iloc[i]
        return torch.tensor([label_val], dtype=torch.long)

    def len(self):
        return len(self.event_data) if self.event_data is not None else 0
    
    def get(self, idx): #get graph data object for event idx
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only = False)
        if self.feature_names:
            data.feature_names = self.feature_names
        return data
    
if __name__ == '__main__':
    dataset = CPDataSet(root = graphsdir) #initialize and process dataset if needed

    print(f"\nDataset loaded successfully!")
    print(f"Number of graphs: {len(dataset)}") #number of events with valid graphs
    print(f"Example graph:\n{dataset[0]}")
    print(f"Node features shape: {dataset[0].x.shape}")
    print(f"Edge index shape: {dataset[0].edge_index.shape}")
