"""
DNN_preprocess.py  — minimal changes to accept an already-built top-features dataset
"""

import h5py
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse, joblib, numpy as np, os

def split_and_save(X_all, y_all, lumi_all, event_all, save_path, train_type):
    """Create splits (Even/Odd or Random) and save in your existing tensor dict format."""
    # Ensure dtypes
    X_all = torch.as_tensor(X_all, dtype=torch.float32)
    y_all = torch.as_tensor(y_all, dtype=torch.long)        # cast to long; you convert to float in the DataLoader
    lumi_all  = torch.as_tensor(lumi_all, dtype=torch.float32)
    event_all = torch.as_tensor(event_all, dtype=torch.long)

    if train_type in ("Even", "Odd"):
        even_mask = (event_all % 2 == 0)
        odd_mask  = ~even_mask

        if train_type == "Even":
            X_train, y_train, lumi_train = X_all[even_mask], y_all[even_mask], lumi_all[even_mask]
            X_test,  y_test,  lumi_test  = X_all[odd_mask],  y_all[odd_mask],  lumi_all[odd_mask]
        else:  # "Odd"
            X_train, y_train, lumi_train = X_all[odd_mask],  y_all[odd_mask],  lumi_all[odd_mask]
            X_test,  y_test,  lumi_test  = X_all[even_mask], y_all[even_mask], lumi_all[even_mask]

        tensors = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'lumi_train': lumi_train, 'lumi_test': lumi_test,
            # optional: keep full arrays for later diagnostics
            'event_train': event_all[(event_all % 2 == 0) if train_type == "Even" else (event_all % 2 == 1)],
            'event_test':  event_all[(event_all % 2 == 1) if train_type == "Even" else (event_all % 2 == 0)],
        }

    elif train_type == "Random":
        # stratified 80/20 random split
        X_np = X_all.numpy(); y_np = y_all.numpy(); lumi_np = lumi_all.numpy()
        X_tr, X_te, y_tr, y_te, lw_tr, lw_te = train_test_split(
            X_np, y_np, lumi_np, test_size=0.2, stratify=y_np, random_state=42
        )
        tensors = {
            'X_train': torch.tensor(X_tr, dtype=torch.float32),
            'X_test':  torch.tensor(X_te, dtype=torch.float32),
            'y_train': torch.tensor(y_tr, dtype=torch.long),
            'y_test':  torch.tensor(y_te, dtype=torch.long),
            'lumi_train': torch.tensor(lw_tr, dtype=torch.float32),
            'lumi_test' : torch.tensor(lw_te, dtype=torch.float32),
        }
    else:
        raise ValueError(f"Unknown train_type={train_type}")

    outdir = os.path.dirname(save_path); os.makedirs(outdir, exist_ok=True)
    torch.save(tensors, save_path)
    print(f"✅ Saved split tensors to {save_path} with keys: {list(tensors.keys())}")

def main(input_path, save_path, train_type, top_features_path=None):
    if top_features_path:
        # ---- NEW: load the MI-selected feature matrix (.pt) and split here ----
        pack = torch.load(top_features_path, weights_only=False)
        X_all = pack['X']                      # already scaled/consistent
        y_all = pack['y']                      # 0/1 labels
        # If your top_features pack didn’t include lumi/event, fallback to HDF5 (rare):
        if 'lumi' in pack and 'event' in pack:
            lumi_all  = pack['lumi']
            event_all = pack['event']
        else:
            # fallback: read from H5 to fetch lumi/event by row alignment
            with h5py.File(input_path) as f:
                df = pd.DataFrame(f['LargeRJet']['1d'][:])
            lumi_all  = df['Lumi_weight'].to_numpy().astype(np.float32)
            event_all = df['EventNumber'].to_numpy().astype(np.int64)

        split_and_save(X_all, y_all, lumi_all, event_all, save_path, train_type)
        return

    # ---- ORIGINAL PATH (unchanged): from H5, scale, then split ----
    with h5py.File(input_path) as f:
        df = pd.DataFrame(f['LargeRJet']['1d'][:])

    if train_type in ('Even', 'Odd'):
        if train_type == 'Even':
            train_df = df[df['EventNumber'] % 2 == 0]
            test_df  = df[df['EventNumber'] % 2 == 1]
        else:
            train_df = df[df['EventNumber'] % 2 == 1]
            test_df  = df[df['EventNumber'] % 2 == 0]

        y_train = (train_df['Lumi_weight'].to_numpy() > 0).astype(np.int64)
        y_test  = (test_df['Lumi_weight'].to_numpy()  > 0).astype(np.int64)
        lumi_train = train_df['Lumi_weight'].copy()
        lumi_test  = test_df['Lumi_weight'].copy()

        X_train_df = train_df.drop(columns=['Lumi_weight','EventNumber','FJ_flavour'])
        X_test_df  = test_df.drop(columns=['Lumi_weight','EventNumber','FJ_flavour'])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df)
        X_test  = scaler.transform(X_test_df)
        joblib.dump(scaler, 'scaler.joblib')

        tensors = {
            'X_train': torch.tensor(X_train, dtype=torch.float32),
            'X_test' : torch.tensor(X_test,  dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.long),
            'y_test' : torch.tensor(y_test,  dtype=torch.long),
            'lumi_train': torch.tensor(lumi_train.to_numpy(), dtype=torch.float32),
            'lumi_test' : torch.tensor(lumi_test.to_numpy(),  dtype=torch.float32),
        }
        outdir = os.path.dirname(save_path); os.makedirs(outdir, exist_ok=True)
        torch.save(tensors, save_path)
        print(f"✅ Saved split tensors to {save_path}")

    elif train_type == 'Random':
        X = df.drop(columns=['Lumi_weight'])
        y = (df['Lumi_weight'].to_numpy() > 0).astype(np.int64)
        lw = df['Lumi_weight'].to_numpy().astype(np.float32)

        X_tr, X_te, y_tr, y_te, lw_tr, lw_te = train_test_split(X, y, lw, test_size=0.2, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        joblib.dump(scaler, 'scaler.joblib')

        tensors = {
            'X_train': torch.tensor(X_tr, dtype=torch.float32),
            'X_test' : torch.tensor(X_te, dtype=torch.float32),
            'y_train': torch.tensor(y_tr, dtype=torch.long),
            'y_test' : torch.tensor(y_te, dtype=torch.long),
            'lumi_train': torch.tensor(lw_tr, dtype=torch.float32),
            'lumi_test' : torch.tensor(lw_te, dtype=torch.float32),
        }
        outdir = os.path.dirname(save_path); os.makedirs(outdir, exist_ok=True)
        torch.save(tensors, save_path)
        print(f"✅ Saved split tensors to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="../data/s2286706/new_Input_CP_Studies_llqq_LinearTerm_20th_October2025.h5")
    parser.add_argument('--save_path',  type=str, default="../data/processed/data_tensors.pt")
    parser.add_argument('--train_type', type=str, default='Even', choices=['Even','Odd','Random'])
    # NEW: path to top_features_*.pt; if provided, we skip H5 and scaling and just split
    parser.add_argument('--top_features_path', type=str, default='../data/processed/top_features_40.pt')
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        save_path=args.save_path,
        train_type=args.train_type,
        top_features_path=args.top_features_path
    )
