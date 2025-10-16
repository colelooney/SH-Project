from GNN import GCN
import torch
from preprocess_graph_data import CPDataSet
from torch_geometric.data import DataLoader
import pandas as pd
import os.path as osp
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np


def main():
    dataset = CPDataSet(root = '../graphdata/CP_Studies_llqq_graphs')

    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    input_size = dataset.num_node_features

    num_graphs = int(len(dataset))
    split_idx = int(0.8 * len(dataset))
    dev_idx = int(0.8 * split_idx)

    train_dev_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    train_dataset = train_dev_dataset[:dev_idx]
    val_dataset = train_dev_dataset[dev_idx:]



    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")


    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    dev_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    hidden_dim = 16
    learning_rate = 0.01

    model = GCN(input_size, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 5e-4)
    criterion = torch.nn.BCELoss()

    num_epochs = 1

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
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), 'ModelsGNN/gnn_model.pth')

    model.eval()
    total_dev_loss = 0 
    all_val_preds = []
    all_val_labels = []
    all_val_probs = []
    all_val_discriminents = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(out, batch.y.float().unsqueeze(1))
            total_dev_loss += loss.item() * batch.num_graphs

            preds = (out > 0.5).long().squeeze()

            all_val_preds.append(preds.cpu())
            all_val_labels.append(batch.y.cpu())
            all_val_probs.append(out.cpu().squeeze())
            discriminant_scores = out.squeeze() - (1 - out).squeeze()
            all_val_discriminents.append(discriminant_scores.cpu())
    
    all_val_preds = torch.cat(all_val_preds)
    all_val_labels = torch.cat(all_val_labels)
    all_val_probs = torch.cat(all_val_probs)
    all_val_discriminents = torch.cat(all_val_discriminents)


    avg_test_loss = total_dev_loss / len(val_dataset)
    accuracy = accuracy_score(all_val_labels, all_val_preds)
    precision = precision_score(all_val_labels, all_val_preds)
    recall = recall_score(all_val_labels, all_val_preds)
    auc = roc_auc_score(all_val_labels, all_val_probs)

    print(f"Average Val Loss: {avg_test_loss:.4f}")
    print(f"VAL Accuracy: {accuracy:.4f}")
    print(f"VAL Precision: {precision:.4f}")
    print(f"VAL Recall: {recall:.4f}")
    print(f"VAL ROC AUC: {auc:.4f}")

    print("\n--- Final Output Inspection ---")
    print("Shape of probabilities_x p_plus tensor:", all_val_probs.shape)
    print("Shape of discriminant_scores tensor:", all_val_discriminents.shape)


    model.eval()
    total_test_loss = 0 
    all_preds = []
    all_labels = []
    all_probs = []
    all_discriminents = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(out, batch.y.float().unsqueeze(1))
            total_test_loss += loss.item() * batch.num_graphs

            preds = (out > 0.5).long().squeeze()

            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())
            all_probs.append(out.cpu().squeeze())
            discriminant_scores = out.squeeze() - (1 - out).squeeze()
            all_discriminents.append(discriminant_scores.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    all_discriminents = torch.cat(all_discriminents)


    avg_test_loss = total_test_loss / len(test_dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    print("\n--- Final Output Inspection ---")
    print("Shape of probabilities_x p_plus tensor:", all_probs.shape)
    print("Shape of discriminant_scores tensor:", all_discriminents.shape)

    np.savez(
        '../graphdata/gnn_discriminant_scores.npz',
        discriminant_scores = all_discriminents.numpy(),
        y_true = all_labels.numpy(),
        y_pred = all_preds.numpy()
    )

    
if __name__ == "__main__":
    main()


    