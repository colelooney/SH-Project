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
from DNN_models import DNN, EarlyStopperAUC
import argparse
from sklearn.metrics import roc_auc_score

def main(tensor_path, model_path, learning_rate, batch_size, num_epochs):

    data_dict = torch.load(tensor_path)
    X_train_tensor = data_dict['X_train']
    # X_val_tensor =  data_dict['X_val']

    y_train_tensor = data_dict['y_train']
    # y_val_tensor = data_dict['y_val']

    input_size = X_train_tensor.shape[1]
    learning_rate = learning_rate
    batch_size = batch_size
    num_epochs = num_epochs

    model = DNN(input_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    early_stopper = EarlyStopperAUC(patience = 5, min_delta = 1e-4)

    

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.float())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # val_dataset = TensorDataset(X_val_tensor, y_val_tensor.float())
    # val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)


    train_losses = []
    rocs = []
    epochs = []
    val_aucs = []

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_probs_list = []
        train_labels_list = []
        for i, (features, labels) in enumerate(train_loader):
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))

            probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()

            train_probs_list.append(probs)
            train_labels_list.append(labels.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_probs = np.concatenate(train_probs_list)
        train_labels = np.concatenate(train_labels_list)
        train_roc = roc_auc_score(train_labels,train_probs)

        train_losses.append(loss.item())
        rocs.append(train_roc)
        epochs.append(epoch)

        # model.eval()
        # val_logits_list = []
        # val_labels_list = []

        # with torch.no_grad():
        #     for i, (features, labels) in enumerate(val_loader):
        #         logits = model(features)
        #         probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

        #         val_logits_list.append(probs)
        #         val_labels_list.append(labels.cpu().numpy())

        #     val_probs = np.concatenate(val_logits_list)
        #     val_labels = np.concatenate(val_labels_list)       

        #     auc_score = roc_auc_score(val_labels,val_probs)
        #     val_aucs.append(auc_score) 
        #     if early_stopper.early_stop(auc_score):
        #         print('Early Stopping due to no improvement in validation loss')
        #         break


        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    #Save the model
    torch.save(model.state_dict(), model_path)

    np.savez(
        '../results/training_results.npz',
        losses =  np.array(train_losses),
        rocs = np.array(rocs),
        epochs = np.array(epochs),
        val_aucs = np.array(val_aucs)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_path',type=str,required=False,default='../data/processed/data_tensors.pt')
    parser.add_argument('--model_path',type=str,required=False,default='ModelsDNN/dnn_model.pth')
    parser.add_argument('--learning_rate',type = float, required= False, default = 0.00017)
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