from GNN import GCN
import torch
from preprocess_graph_data import CPDataSet, train_dataset, test_dataset
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np

def main():
    """
    Main function to test the GCN model
    """

    # dataset = CPDataSet(root = '../graphdata/CP_Studies_llqq_graphs')

    input_size = train_dataset.num_node_features
    hidden_dim = 16
    learning_rate = 0.01
    batch_size = 64

    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    criterion = torch.nn.BCELoss()


    model = GCN(input_size, hidden_dim)
    model.load_state_dict(torch.load('ModelsGNN/gnn_model.pth'))
    

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

    print("\n--- Final Output Inspection ---")
    print("Shape of probabilities_x p_plus tensor:", all_probs.shape)
    print("Shape of discriminant_scores tensor:", all_discriminents.shape)

    np.savez(
        './data/dnn_discriminant_scores_and_lumi_weights.npz',
        discriminant_scores = all_discriminents.numpy(),
        y_true = all_labels.numpy(),
        y_pred = all_preds.numpy()
    )


