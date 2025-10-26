"""
Cole Looney 26/10/2025

GNN_preprocess.py

Process 2d h5 data into PyG Data Objects for Graph Neural Network to Use

arguments:
--hfivesdir: directory path to h5 file
--graphdir: directory path to where processed data objects are stored
--lepton_only: If True, only keep constituents on leptons, default is True

"""


import h5py
import numpy as np
import pandas as pd
import torch
import os.path as osp
from torch_geometric.data import Dataset, Data
import argparse

class CPDataSet(Dataset):
    def __init__(self,root, hfivesdir, lepton_only = True, transform = None, pre_transform = None, pre_filter = None):
        self.event_data = None #store as attribute to access later
        self.feature_names = None
        self.hfivesdir = hfivesdir
        self.lepton_only = lepton_only
        super().__init__(root,transform,pre_transform,pre_filter) #calls process if data not processed

        if osp.exists(self.processed_paths[1]): #load event data info if it exists
            self.event_data = torch.load(self.processed_paths[1], weights_only = False)
        if osp.exists(self.processed_paths[2]):
            self.feature_names = torch.load(self.processed_paths[2], weights_only = False)
    
    @property
    def raw_file_names(self):
        return [osp.basename(self.hfivesdir)]
    
    @property
    def processed_file_names(self):
        return ['data_0.pt', 'event_data_info.pt', 'feature_names.pt']
    
    def download(self):
        pass


    def process(self):
        # load raw data
        with h5py.File(self.hfivesdir, 'r') as f:
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
        graph_save_idx = 0
        processed_event_data = []
        for i in range(num_events):
            mask = valid_mask[i] #mask for valid constituents in event i
            valid_nodes = constant_features[i,mask,:] #get valid constituents for event i
            
            if self.lepton_only:
                lepton_mask_for_event = valid_nodes[:, isLep_idx] == 1
                valid_nodes = valid_nodes[lepton_mask_for_event]

            if valid_nodes.shape[0] < 2: #skip if not enough nodes to create an edge
                continue


            node_feats = self.__get_node_features(valid_nodes) #get node features tensor
            edge_index = self.__get_edge_index(valid_nodes=valid_nodes.shape[0]) #get edge index tensor
            label = self._get_labels(i, event_data_local) #get label tensor
            lumi_weight_tensor = torch.tensor([event_data_local['Lumi_weight'].iloc[i]],dtype = torch.float)

            data = Data(x = node_feats, edge_index = edge_index, y = label, lumi_weight = lumi_weight_tensor) #create PyG Data object

            torch.save(data, osp.join(self.processed_dir, f'data_{graph_save_idx}.pt')) #save graph data object
            processed_event_data.append(df_1d.iloc[i])
            graph_save_idx += 1

            if (i + 1) % 5000 == 0:
                print(f'Processed {i+1}/{num_events} events.')

        final_event_df = pd.DataFrame(processed_event_data)
        print(f"\nProcessing complete. Created {len(final_event_df)} valid graphs.")
        print("Saving filtered event-level information...")
        torch.save(final_event_df, self.processed_paths[1])

    def __get_node_features(self, valid_nodes):
        return torch.tensor(valid_nodes, dtype = torch.float) #all features for now
    
    def __get_edge_index(self, valid_nodes):
        # connect all nodes => fully connected graph
        #Since only two leptons decay will always be [1,0] and [0,1]
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
    
def data_splitter(graphsdir,test_size,save_path):
    dataset = CPDataSet(root = graphsdir)

    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    input_size = dataset.num_node_features

    split_idx = int((1-2*test_size) * len(dataset))

    train_dataset = dataset[:split_idx]
    test_val_dataset = dataset[split_idx:]

    dev_idx = int(0.5 * len(test_val_dataset))
    test_dataset = test_val_dataset[:dev_idx]
    val_dataset = test_val_dataset[dev_idx:]

    dataset_dict = {
        'train_dataset':train_dataset,
        'val_dataset':val_dataset,
        'test_dataset':test_dataset
    }

    torch.save(dataset_dict,save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hfivesdir',type=str,required=False, default = '../data/s2286706/new_Input_CP_Studies_llqq_QuadraticTerm_20th_October2025.h5')
    parser.add_argument('--graphdir',type=str,default="../graphdata/CP_Studies_llqq_graphs_20th_October_Quadratic", required = False)
    parser.add_argument('--lepton_only',type = bool,default = True, required = False)
    parser.add_argument('--test_size',type=float,required=False,default=0.25)
    parser.add_argument('--save_path',type=str,required=False, default = '../graphdata/dataset_dict')

    args = parser.parse_args()
    dataset = CPDataSet(root = args.graphsdir,
                         hfivesdir=args.hfivesdir,
                         lepton_only = args.lepton_only) #initialize and process dataset if needed

    print(f"\nDataset loaded successfully!")
    print(f"Number of graphs: {len(dataset)}") #number of events with valid graphs
    print(f"Example graph:\n{dataset[0]}")
    print(f"Node features shape: {dataset[0].x.shape}")
    print(f"Edge index shape: {dataset[0].edge_index.shape}")

    print(f'\nsplitting datasets with test size {args.test_size}')
    data_splitter(graphsdir=args.graphdir,
                  test_size=args.test_size,
                  save_path=args.save_path)
    print('\nGraph data processing complete')