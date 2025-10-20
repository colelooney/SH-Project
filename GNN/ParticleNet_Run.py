from GNN import ParticleNet 
import torch
from preprocess_graph_data import CPDataSet
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np


def main():
    dataset = CPDataSet(root = '../graphdata/CP_Studies_llqq_graphs')

    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    input_size = dataset.num_node_features

    num_graphs = int(len(dataset))
    split_idx = int(0.5 * len(dataset))

    train_dataset = dataset[:split_idx]
    test_val_dataset = dataset[split_idx:]

    dev_idx = int(0.5 * len(test_val_dataset))
    test_dataset = test_val_dataset[:dev_idx]
    val_dataset = test_val_dataset[dev_idx:]




    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")

    train_labels = torch.cat([data.y for data in train_dataset])
    print(train_labels)
    num_class_0 = (train_labels == 0).sum()
    num_class_1 = (train_labels == 1).sum()
    print(f"Number of class 0 in training set: {num_class_0}")
    print(f"Number of class 1 in training_set: {num_class_1}")




    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    dev_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    model = ParticleNet(
        kernel_sizes = [64, 128, 256],
        fc_size = 128,
        dropout = 0.3,
        k = 16,
        node_feat_size = dataset.num_node_features,
        num_classes = 2
    )

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 50

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(1,num_epochs+1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")

        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in dev_loader:
                out = model(batch)
                loss = criterion(out,batch.y)
                total_val_loss += loss.item() * batch.num_graphs

                probs = out.softmax(dim=1)[:, 1]
                preds = (probs > 0.5).long()

                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())
                all_probs.append(probs.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)

        avg_val_loss = total_val_loss / len(val_dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        
        print(f"Epoch: {epoch:03d}, Train Loss: {avg_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4f}, Val AUC: {auc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'ModelsGNN/particlenet_best_model.pth')
            patience_counter = 0
            print("  -> Validation loss improved, saving model.")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  -> Stopping early after {patience} epochs with no improvement.")
            break
    
    torch.save(model.state_dict(), 'ModelsGNN/gnn_model.pth')

    model.eval()
    total_dev_loss = 0 
    all_val_preds = []
    all_val_labels = []
    all_val_probs = []
    all_val_discriminents = []
    all_val_lumi_weights = []
    with torch.no_grad():
        for batch in dev_loader:
            out = model(batch)

            loss = criterion(out, batch.y)
            total_dev_loss += loss.item() * batch.num_graphs

            probs = out.softmax(dim=1)[:, 1]
            preds = (probs > 0.5).long()

            all_val_preds.append(preds.cpu())
            all_val_labels.append(batch.y.cpu())
            all_val_probs.append(probs.cpu())
            discriminant_scores = probs - (1 - probs)
            all_val_lumi_weights.append(batch.lumi_weight.cpu())
            all_val_discriminents.append(discriminant_scores.cpu())
    
    all_val_preds = torch.cat(all_val_preds)
    all_val_labels = torch.cat(all_val_labels)
    all_val_probs = torch.cat(all_val_probs)
    all_val_discriminents = torch.cat(all_val_discriminents)
    all_val_lumi_weights = torch.cat(all_val_lumi_weights)


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

    np.savez(
        '../graphdata/gnn_discriminant_scores_validation.npz',
        discriminant_scores = all_val_discriminents.numpy(),
        y_true = all_val_labels.numpy(),
        y_pred = all_val_preds.numpy(),
        lumi_weights = all_val_lumi_weights.numpy()
    )


    model.eval()
    total_test_loss = 0 
    all_preds = []
    all_labels = []
    all_probs = []
    all_discriminents = []
    all_lumi_weights = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)

            loss = criterion(out, batch.y)
            total_test_loss += loss.item() * batch.num_graphs

            probs = out.softmax(dim = 1)[:,1]
            preds = (probs > 0.5).long()

            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())
            all_probs.append(probs.cpu())
            discriminant_scores = probs - (1 - probs)
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

    print("\n--- Final Output Inspection ---")
    print("Shape of probabilities_x p_plus tensor:", all_probs.shape)
    print("Shape of discriminant_scores tensor:", all_discriminents.shape)

    np.savez(
        '../graphdata/gnn_discriminant_scores.npz',
        discriminant_scores = all_discriminents.numpy(),
        y_true = all_labels.numpy(),
        y_pred = all_preds.numpy(),
        lumi_weights = all_lumi_weights.numpy()
    )

    
if __name__ == "__main__":
    main()


    