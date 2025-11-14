
"""
Cole Looney

DNN_preprocess.py

Process 1d decay data from h5py to PyTorch tensors

arguments:
--input_path: location of h5 file
--save_path: path to save dictionary containing split dataset tensors
--train_type: split dataset by event number into Even/ODD (ODD/Even) train/test or random train/test
"""

import h5py
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import joblib
import numpy as np
import os


def main(input_path, save_path, train_type):
    with h5py.File(input_path) as f:
        df = pd.DataFrame(f['LargeRJet']['1d'][:])

    features_to_drop = ['EventNumber', 'FJ_flavour', 'Lumi_weight','Type']

    
    if train_type == 'Even':
        train_dev = df[df['EventNumber'] % 2 ==0]
        split_idx = int(0.8*len(train_dev))
        # val = train_dev[split_idx:]
        # train = train_dev[:split_idx]
        train = train_dev
        test = df[df['EventNumber'] % 2 == 1]

        lumi_train =  train['Lumi_weight'].copy()
        # lumi_val = val['Lumi_weight'].copy()
        lumi_test = test['Lumi_weight'].copy()

        y_train = train['Lumi_weight'].copy()
        X_train = train.drop(columns=features_to_drop)

        # y_val = val['Lumi_weight'].copy()
        # X_val = val.drop(columns=['Lumi_weight','EventNumber','FJ_flavour'])

        y_test = test['Lumi_weight'].copy()
        X_test = test.drop(columns=features_to_drop)

        y_train[y_train>0] = 1
        y_train[y_train<0] = 0

        # y_val[y_val>0] = 1
        # y_val[y_val<0] = 0

        y_test[y_test>0] = 1
        y_test[y_test<0] = 0

        y_train = np.array(y_train)
        # y_val = np.array(y_val)
        y_test = np.array(y_test)

    elif train_type == 'Odd':
        train_dev = df[df['EventNumber'] % 2 ==1]
        # split_idx = int(0.8*len(train_dev))
        # val = train_dev[split_idx:]
        # train = train_dev[:split_idx]
        train = train_dev
        test = df[df['EventNumber'] % 2 == 0]

        lumi_train =  train['Lumi_weight'].copy()
        # lumi_val = val['Lumi_weight'].copy()
        lumi_test = test['Lumi_weight'].copy()

        y_train = train['Lumi_weight'].copy()
        X_train = train.drop(columns=features_to_drop)

        # y_val = val['Lumi_weight'].copy()
        # X_val = val.drop(columns=features_to_drop)

        y_test = test['Lumi_weight'].copy()
        X_test = test.drop(columns=features_to_drop)

        y_train[y_train>0] = 1
        y_train[y_train<0] = 0

        # y_val[y_val>0] = 1
        # y_val[y_val<0] = 0

        y_test[y_test>0] = 1
        y_test[y_test<0] = 0

        y_train = np.array(y_train)
        # y_val = np.array(y_val)
        y_test = np.array(y_test)

    elif train_type == 'Random':
        X = df.drop(columns=features_to_drop)
        y = df['Lumi_weight'].copy()
        lumi_weights =  df['Lumi_weight'].copy()

        #Assign labels to y
        y[y>0] = 1
        y[y<0] = 0

        #Split data into train and validation + testing set
        X_train, X_test_val, y_train, y_test_val, lumi_train, lumi_test_val = train_test_split(X, y,lumi_weights, test_size=0.2, random_state=42)

        #splti validation and testing into two equally sized sets
        X_val, X_test, y_val, y_test, lumi_val, lumi_test = train_test_split(X_test_val,y_test_val,lumi_test_val,test_size = 0.5,stratify=  y_test_val, random_state = 42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    #save scaler for reproducibility
    joblib.dump(scaler, 'scaler.joblib')

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    # X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    # y_val_tensor=torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    lumi_train_tensor=torch.tensor(lumi_train.to_numpy(), dtype=torch.float32)
    # lumi_val_tensor = torch.tensor(lumi_val.to_numpy(), dtype=torch.float32)
    lumi_test_tensor = torch.tensor(lumi_test.to_numpy(), dtype=torch.float32)

    #bundle tensors into a dictionary
    tensors = {
        'X_train': X_train_tensor,
        # 'X_val': X_val_tensor,
        'X_test': X_test_tensor,

        'y_train': y_train_tensor,
        # 'y_val': y_val_tensor,
        'y_test': y_test_tensor,

        'lumi_train': lumi_train_tensor,
        # 'lumi_val': lumi_val_tensor,
        'lumi_test': lumi_test_tensor
    }

    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(tensors, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path',required = False, type = str, default = "../data/s2286706/new_Input_CP_Studies_llqq_LinearTerm_20th_October2025.h5")
    parser.add_argument('--save_path', required = False, type = str, default = "../data/processed/data_tensors.pt")
    parser.add_argument('--train_type',required = False, default = 'Even',type = str, choices = ['Even','Odd','Random'])
    args = parser.parse_args()

    main(input_path = args.input_path,
         save_path = args.save_path,
         train_type = args.train_type)
