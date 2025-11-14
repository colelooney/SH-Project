"""
Cole Looney — 2025

DNN_feature_stability_test.py
-----------------------------------
Test robustness of feature count (N = 20, 25, 32)
using multiple random seeds to determine the
optimal number of features for the final DNN model.

Loads top-N ranked features by mutual information,
trains the DNN, and reports mean±std validation AUC.
Plots results with error bars.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from DNN_models import DNN
import h5py
import random
import os
import matplotlib.pyplot as plt

# -----------------------
# CONFIGURATION
# -----------------------
relative_path = '../data/s2286706/new_Input_CP_Studies_llqq_LinearTerm_20th_October2025.h5'
feature_rank_csv = '../results/mutual_information_base.csv'  # from MI calculation
output_csv = '../results/feature_stability_auc.csv'
output_plot = '../plots/feature_stability_auc.png'

N_FEATURES = [18, 20,25, 32]
SEEDS = [0, 1, 2, 3, 4]
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# LOAD DATA + FEATURE RANKING
# -----------------------
with h5py.File(relative_path) as f:
    df = pd.DataFrame(f['LargeRJet']['1d'][:])

feature_ranking = pd.read_csv(feature_rank_csv)
feature_ranking = feature_ranking.sort_values(by='MutualInfo', ascending=False)

# -----------------------
# PREPROCESS FUNCTION
# -----------------------
def preprocess(df, feature_list):
    df = df.copy()
    X = df[feature_list]
    y = df['Lumi_weight'].copy()
    y[y > 0] = 1
    y[y < 0] = 0
    y = y.to_numpy()

    # even/odd split
    train_df = df[df['EventNumber'] % 2 == 0]
    test_df = df[df['EventNumber'] % 2 == 1]

    X_train = train_df[feature_list].values
    X_test = test_df[feature_list].values
    y_train = (train_df['Lumi_weight'].values > 0).astype(int)
    y_test = (test_df['Lumi_weight'].values > 0).astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

# -----------------------
# TRAIN FUNCTION
# -----------------------
def train_one_run(X_train, y_train, X_test, y_test, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = DNN(X_train.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )

    for epoch in range(EPOCHS):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

    # evaluate
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test.to(DEVICE))).cpu().numpy().flatten()
    auc = roc_auc_score(y_test.numpy(), probs)
    return auc

# -----------------------
# MAIN LOOP
# -----------------------
results = []

for n in N_FEATURES:
    top_features = feature_ranking['Feature'].iloc[:n].tolist()
    X_train, X_test, y_train, y_test = preprocess(df, top_features)

    aucs = []
    for seed in SEEDS:
        auc = train_one_run(X_train, y_train, X_test, y_test, seed)
        aucs.append(auc)
        print(f"N={n}, seed={seed}, AUC={auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    results.append({'N_features': n, 'Mean_AUC': mean_auc, 'Std_AUC': std_auc})

    print(f"✅ N={n} | Mean AUC={mean_auc:.4f} ± {std_auc:.4f}")

# -----------------------
# SAVE RESULTS
# -----------------------
results_df = pd.DataFrame(results)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
os.makedirs(os.path.dirname(output_plot), exist_ok=True)
results_df.to_csv(output_csv, index=False)

# -----------------------
# PLOT RESULTS
# -----------------------
plt.figure(figsize=(8,5))
plt.errorbar(
    results_df['N_features'], 
    results_df['Mean_AUC'], 
    yerr=results_df['Std_AUC'],
    fmt='o-', capsize=5, lw=2, color='navy', ecolor='lightblue'
)
best_idx = results_df['Mean_AUC'].idxmax()
best_n = results_df.loc[best_idx, 'N_features']
best_auc = results_df.loc[best_idx, 'Mean_AUC']

plt.axvline(best_n, color='orange', linestyle='--', label=f'Best N={best_n}')
plt.title("Validation AUC vs Number of Top Features", fontsize=13)
plt.xlabel("Number of Features (N)", fontsize=12)
plt.ylabel("Mean AUC ± Std", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.show()

print("\n--- Summary ---")
print(results_df)
print(f"\nBest configuration: N={best_n}, Mean AUC={best_auc:.4f}")
