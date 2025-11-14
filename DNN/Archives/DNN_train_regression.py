"""
Cole Looney 25/10/2025

DNN_train.py

Train DNN model on high-level observables with early stopping

arguments:
--tensor_path: directory path to where tensor dictionary is stored
--model_path: path to save trained model params for evaluation
--learning_rate: model learning rate
--batch_size: pytorch dataloader batch size
--num_epochs: number of maximum training epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from DNN_models import DNN, EarlyStopper
import argparse

def main(tensor_path, model_path, learning_rate, batch_size, num_epochs):

    data_dict = torch.load(tensor_path)
    X_train_tensor = data_dict['X_train']
    # X_val_tensor =  data_dict['X_val']

    y_train_tensor = data_dict['y_train']
    # y_val_tensor = data_dict['y_val']

    lumi_train_tensor = data_dict['lumi_train']

    input_size = X_train_tensor.shape[1]
    learning_rate = learning_rate
    batch_size = batch_size
    num_epochs = num_epochs

    model = DNN(input_size)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # early_stopper = EarlyStopper(patience = 3, min_delta = .1)

    
    lumi_mean = lumi_train_tensor.mean()
    lumi_std = lumi_train_tensor.std()
    lumi_train_tensor = (lumi_train_tensor - lumi_mean) / lumi_std

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.float(),lumi_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # val_dataset = TensorDataset(X_val_tensor, y_val_tensor.float())
    # val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        for i, (features, labels,lumi_weight) in enumerate(train_loader):
            outputs = model(features).squeeze()
            loss = criterion(outputs, lumi_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # model.eval()
        # with torch.no_grad():
        #     total_val_loss = 0
        #     for i, (features, labels) in enumerate(val_loader):
        #         probs_pplus = model(features)
        #         total_val_loss += criterion(probs_pplus,labels.unsqueeze(1))
        #     if early_stopper.early_stop(total_val_loss):
        #         print('Early Stopping due to no improvement in validation loss')
        #         break


        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    #Save the model
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_path',type=str,required=False,default='../data/processed/data_tensors.pt')
    parser.add_argument('--model_path',type=str,required=False,default='ModelsDNN/dnn_model.pth')
    parser.add_argument('--learning_rate',type = float, required= False, default = 0.0001)
    parser.add_argument('--batch_size', type = int, required=False, default = 128)
    parser.add_argument('--num_epochs',type = int,required=False, default = 20)

    args = parser.parse_args()
    main(
        tensor_path=args.tensor_path,
        model_path=args.model_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )