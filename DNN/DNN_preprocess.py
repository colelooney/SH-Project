
"""
Cole Looney

Data Preprocessing Script
"""

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import joblib


def main(input_path):
    with h5py.File(input_path) as f:
        df = pd.DataFrame(f['LargeRJet']['1d'][:])

    X = df.drop(columns=['Lumi_weight'])
    y = df['Lumi_weight'].copy()
    lumi_weights =  df['Lumi_weight'].copy()

    #Assign labels to y
    y[y>0] = 1
    y[y<0] = 0

    #Split data into train and validation + testing set
    X_train, X_test_val, y_train, y_test_val, lumi_train, lumi_test_val = train_test_split(X, y,lumi_weights, test_size=0.2, random_state=42, stratify = y)

    #splti validation and testing into two equally sized sets
    X_val, X_test, y_val, y_test, lumi_val, lumi_test = train_test_split(X_test_val,y_test_val,lumi_test_val,test_size = 0.5,stratify=  y, random_state = 42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    #save scaler for reproducibility
    joblib.dump(scaler, 'scaler.joblib')

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor=torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    lumi_train_tensor=torch.tensor(lumi_train, dtype=torch.float32)
    lumi_val_tensor = torch.tensor(lumi_val, dtype=torch.float32)
    lumi_test_tensor = torch.tensor(lumi_test, dtype=torch.float32)

    #bundle tensors into a dictionary
    tensors = {
        'X_train': X_train_tensor,
        'X_val': X_val_tensor,
        'X_test': X_test_tensor,

        'y_train': y_train_tensor,
        'y_val': y_val_tensor,
        'y_test': y_test_tensor,

        'lumi_train': lumi_train_tensor,
        'lumi_val': lumi_val_tensor,
        'lumi_test': lumi_test_tensor
    }

    torch.save(tensors, '../data/processed/data_tensors.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser

    parser.add_argument('--input_path',required = False, type = str, default = '../data/s2286706/new_Input_CP_Studies_llqq_LinearTerm_20th_October2025.h5')

    args = parser.parse_args()
    main(args.input_path)
