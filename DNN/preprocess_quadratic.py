"""
Cole Looney

preprocess_quadratic.py

Process quadratic term data from h5py to PyTorch tensors

arguments:
--input_path: location of h5 file
--save_path: path to save dictionary containing split dataset tensors
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


def main(input_path, save_path):
    with h5py.File(input_path) as f:
        df = pd.DataFrame(f['LargeRJet']['1d'][:])

    features_to_drop = ['EventNumber', 'FJ_flavour', 'Lumi_weight','Type','NegLep_Eta',
                        'Lep_pT_balance','FJ_mass','FJ_E','FJ_phi','LeadingSubJet_Phi',
                        'NegLep_E','PosLep_Eta','SubLeadingSubJet_pT','Vlep_E','Vlep_mass',
                        'Vlep_phi','cosThetaStar','costheta1']

    

    #load scaler from linear term preprocessing
    scaler = joblib.load('scaler.joblib')

    quad_lumi = df['Lumi_weight'].copy()
    quad_train = df.drop(columns= features_to_drop)
    quad_train = scaler.transform(quad_train)

    quad_tensor = torch.tensor(quad_train,dtype=torch.float32)
    quad_lumi_tensor = torch.tensor(quad_lumi.to_numpy(),dtype=torch.float32)
    
    y_test = (quad_lumi.to_numpy() >0).astype(np.int64)
    y_test_tensor = torch.tensor(y_test,dtype = torch.long)

    #bundle tensors into a dictionary, used to match DNN_eval expectation
    tensors = {
        'X_test': quad_tensor,
        'lumi_test': quad_lumi_tensor,
        'y_test': y_test_tensor

    }

    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(tensors, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path',required = False, type = str, default = "../data/s2286706/new_Input_CP_Studies_llqq_QuadraticTerm_20th_October2025.h5")
    parser.add_argument('--save_path', required = False, type = str, default = "../data/processed/data_tensors_quad.pt")
    args = parser.parse_args()

    main(input_path = args.input_path,
         save_path = args.save_path
         )
