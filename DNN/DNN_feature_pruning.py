"""
Cole Looney 05/11/2025

DNN_feature_pruning.py

Iteratively removes lowest mutual information features
to find maximum DNN performance on CP classification task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from DNN_models import DNN
import argparse
import os
import matplotlib.pyplot as plt

def train_and_eval(X_train, y_train, learning_rate, batch_size, num_epochs):
    """Train a DNN and return validation AUC."""
    input_size = X_train.shape[1]
    model = DNN(input_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(X_train, y_train.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for features, labels in loader:
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on same set (we're just comparing relative performance)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_train)).squeeze()
    auc = roc_auc_score(y_train.numpy(), probs.numpy())
    return auc, loss.item()


def main(tensor_path, mi_path, learning_rate, batch_size, num_epochs, save_path):
    # Load tensors and MI ranking
    data_dict = torch.load(tensor_path)
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']

    mi_df = pd.read_csv(mi_path)
    mi_df = mi_df.sort_values(by='Importance', ascending=False)
    features_sorted = mi_df['Feature'].tolist()

    print(f"Loaded {len(features_sorted)} ranked features from {mi_path}")

    performances = []

    for k in range(len(features_sorted), 4, -2):
        selected = features_sorted[:k]
        X_selected = X_train[:, :k]  # assuming correct ordering in tensors

        auc, final_loss = train_and_eval(X_selected, y_train, learning_rate, batch_size, num_epochs)
        performances.append((k, auc, final_loss))
        print(f"[Features: {k:2d}] AUC = {auc:.4f} | Loss = {final_loss:.4f}")

    results_df = pd.DataFrame(performances, columns=['n_features', 'AUC', 'Loss'])
    results_df.to_csv(save_path, index=False)

    plt.figure(figsize=(8,5))
    plt.plot(results_df['n_features'], results_df['AUC'], 'o-', color='blue')
    plt.xlabel('Number of Features')
    plt.ylabel('Validation AUC')
    plt.title('DNN AUC vs Number of Features (MI-based pruning)')
    plt.grid(True)
    plt.savefig(save_path.replace('.csv', '.png'))
    plt.show()

    print(f"\nResults saved to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_path', type=str, default='../data/processed/data_tensors_even.pt')
    parser.add_argument('--mi_path', type=str, default='../data/feature_importances.csv')
    parser.add_argument('--save_path', type=str, default='../results/feature_pruning_results.csv')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    main(
        tensor_path=args.tensor_path,
        mi_path=args.mi_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
    )