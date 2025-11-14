"""
Cole Looney — 05/11/2025

optuna_tune_dnn.py

Hyperparameter optimization for DNN classifier using Optuna.
Evaluates model performance via validation AUC.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
import numpy as np
import optuna
import argparse
from DNN_models import DNN

# -------------------------
# Objective Function
# -------------------------
def objective(trial, tensor_path):

    # ---- Load data ----
    data_dict = torch.load(tensor_path)
    X = data_dict['X_train']
    y = data_dict['y_train'].float()

    # ---- Create train/val split ----
    n_total = len(X)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        TensorDataset(X, y), [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # ---- Sample hyperparameters ----
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128])
    # num_layers = trial.suggest_int("num_layers", 2, 8)
    lr = trial.suggest_loguniform("lr", 0.9e-4, 1.1e-4)
    # batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.2)
    activation_name = trial.suggest_categorical("activation", ["LeakyReLU", "ReLU"])

    # ---- Define model dynamically ----
    layers = []
    input_size = X.shape[1]
    current_size = input_size
    act_fn = nn.LeakyReLU(0.01) if activation_name == "LeakyReLU" else nn.ReLU()
    batch_size = 128

    for _ in range(4):
        layers.append(nn.Linear(current_size, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act_fn)
        layers.append(nn.Dropout(dropout_rate))
        current_size = hidden_dim

    layers.append(nn.Linear(hidden_dim, 1))
    model = nn.Sequential(*layers)

    # ---- Train/Validation Setup ----
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---- Training Loop ----
    model.train()
    for epoch in range(10):  # keep short for tuning speed
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.unsqueeze(1))
            loss.backward()
            optimizer.step()

    # ---- Validation AUC ----
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds.append(probs)
            truths.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    auc = roc_auc_score(truths, preds)
    trial.report(auc, step=0)

    # ---- Early Pruning ----
    if trial.should_prune():
        raise optuna.TrialPruned()

    return auc


# -------------------------
# Main Tuning Entry
# -------------------------
def main(tensor_path, n_trials, study_name):
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(lambda trial: objective(trial, tensor_path), n_trials=n_trials)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  AUC: {trial.value:.4f}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")

    # Save study for later inspection
    optuna.save_study(study, f"{study_name}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_path", type=str, default="../data/processed/data_tensors_evenval.pt")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="dnn_hyperparam_opt")
    args = parser.parse_args()

    main(args.tensor_path, args.n_trials, args.study_name)
