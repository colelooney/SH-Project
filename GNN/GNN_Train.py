"""
Cole Looney 26/10/2025

GNN_train.py

Train Graph Network with Early Stopping

arguments:
--dict_path: path to dictionary containing datasets
--hidden_dim: number of hidden dimensions in model
--num_epochs: maximum number of training epochs
--learning_rate: model learning rate
--batch_size: DataLoader batch size
"""
from GNN import GCN, EarlyStopper
import torch
from torch_geometric.data import DataLoader
import numpy as np
import argparse
import os
from GNN_preprocess import CPDataSet

def main(dict_path,batch_size,hidden_dim,learning_rate,num_epochs):
    data_dict = torch.load(dict_path, weights_only = False)

    train_dataset = data_dict['train_dataset']
    val_dataset = data_dict['val_dataset']
    input_size = data_dict['input_size']

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # train_labels = torch.cat([data.y for data in train_dataset])
    # num_class_0 = (train_labels == 0).sum()
    # num_class_1 = (train_labels == 1).sum()
    # print(f"Number of class 0 in training set: {num_class_0}")
    # print(f"Number of class 1 in training_set: {num_class_1}")

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    dev_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    model = GCN(input_size, hidden_dim)
    early_stopper = EarlyStopper(patience = 10, min_delta = 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    training_losses = []
    training_epoch = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            out = model(batch.x,batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.float().unsqueeze(1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_dataset)
        training_losses.append(avg_loss)
        training_epoch.append(epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")

        #check validation loss for early stopping
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in dev_loader:
                out = model(batch.x,batch.edge_index,batch.batch)
                loss = criterion(out,batch.y.float().unsqueeze(1))

                total_val_loss += loss.item() * batch.num_graphs

            validation_losses.append(total_val_loss)
            if early_stopper.early_stop(total_val_loss):
                break

    os.makedirs('ModelsGNN', exist_ok=True)
    torch.save(model.state_dict(), 'ModelsGNN/gnn_model_res.pth')

    np.savez('../graphdata/training_loss',
             training_loss = np.array(training_losses),
             epoch = np.array(training_epoch),
             validation_loss=np.array(validation_losses)
             )
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path',type=str,default='../graphdata/dataset_dict.pt',required=False)
    parser.add_argument('--hidden_dim',type=int,default=256,required=False)
    parser.add_argument('--batch_size',default = 128, type = int, required = False)
    parser.add_argument('--learning_rate',type=float,default=1e-4,required=False)
    parser.add_argument('--num_epochs',type=int,default=20,required=False)
    args = parser.parse_args()
    main(dict_path=args.dict_path,
         hidden_dim=args.hidden_dim,
         batch_size=args.batch_size,
         learning_rate=args.learning_rate,
         num_epochs=args.num_epochs)


    