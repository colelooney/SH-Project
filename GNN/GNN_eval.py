"""
Cole Looney 26/10/2025

DNN_eval.py

Evaluate trained GNN model on test data

arguments:
--tensor_path: path to where tensor dictionary from DNN_preprocess.py is located
--model_path: path to trained model dictionary
--numpy_save_filename: filename of numpy array storing model outputs
"""

import torch
from sklearn.metrics import classification_report
import numpy as np
from GNN import GCN
import joblib
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score,precision_score, recall_score, roc_auc_score
from torch_geometric.data import DataLoader
from GNN_preprocess import CPDataSet

def main(dict_path, model_path,batch_size,hidden_dim, save_path):
    data_dict = torch.load(dict_path,weights_only = False)

    test_dataset = data_dict['test_dataset']

    input_size = test_dataset[0].x.shape[1]
    model = GCN(input_size,hidden_dim)
    model.load_state_dict(torch.load(model_path))

    criterion = torch.nn.BCEWithLogitsLoss()

    test_loader = DataLoader(test_dataset,batch_size = batch_size, shuffle = False)

    model.eval()
    total_test_loss = 0 
    all_preds = []
    all_labels = []
    all_probs = []
    all_discriminents = []
    all_lumi_weights = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(out, batch.y.float().unsqueeze(1))
            total_test_loss += loss.item() * batch.num_graphs

            out = torch.sigmoid(out)
            preds = (out > 0.5).long().squeeze()

            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())
            all_probs.append(out.cpu().squeeze())
            discriminant_scores = out.squeeze() - (1 - out).squeeze()
            all_lumi_weights.append(batch.lumi_weight.cpu())
            all_discriminents.append(discriminant_scores.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    all_discriminents = torch.cat(all_discriminents)
    all_lumi_weights = torch.cat(all_lumi_weights)


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

    print(f"Classification Report: {classification_report(all_labels,all_preds)}")

    print("\n--- Final Output Inspection ---")
    print("Shape of probabilities_x p_plus tensor:", all_probs.shape)
    print("Shape of discriminant_scores tensor:", all_discriminents.shape)

    np.savez(
        save_path,
        discriminant_scores = all_discriminents.numpy(),
        y_true = all_labels.numpy(),
        y_pred = all_preds.numpy(),
        lumi_weights = all_lumi_weights.numpy()
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path',type = str, required=False, default ='../graphdata/dataset_dict.pt')
    parser.add_argument('--model_path',type=str,required=False,default='ModelsGNN/gnn_model.pth')
    parser.add_argument('--batch_size', type=int,required=False,default=128)
    parser.add_argument('--hidden_dim',type=int,required=False,default=128)
    parser.add_argument('--save_path',type=str,default='../graphdata/gnn_test_results.npz',required=False)

    args=parser.parse_args()

    main(dict_path=args.dict_path,
         model_path=args.model_path,
         batch_size=args.batch_size,
         hidden_dim=args.hidden_dim,
         save_path=args.save_path)