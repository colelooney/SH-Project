"""
Cole Looney 25/10/2025

DNN_eval.py

Evaluate trained DNN model on test data

arguments:
--tensor_path: path to where tensor dictionary from DNN_preprocess.py is located
--model_path: path to trained model dictionary
--numpy_save_filename: filename of numpy array storing model outputs
"""


import torch
from sklearn.metrics import classification_report
import numpy as np
from DNN_models import DNN
import joblib
import argparse
from datetime import datetime

def main(tensor_path, model_path, save_path):
    data_dict = torch.load(tensor_path)

    X_test_tensor = data_dict['X_test']
    y_test_tensor = data_dict['y_test']
    lumi_test_tensor = data_dict['lumi_test']

    input_size = X_test_tensor.shape[1]
    model = DNN(input_size)
    model.load_state_dict(torch.load(model_path))

    scaler = joblib.load('scaler.joblib')

    X_test_tensor = scaler.transform(X_test_tensor) #to accomodate unseen, external test data
    X_test_tensor = torch.tensor(X_test_tensor, dtype = torch.float32)

    model.eval()
    with torch.no_grad():
        test_prob = model(X_test_tensor)
        p_plus = test_prob.squeeze() #make 1d
        p_minus = 1 - p_plus
        discriminant_scores = p_plus - p_minus
        predicted_labels = (test_prob > 0.5).long().squeeze()

        y_true = y_test_tensor.numpy()
        y_pred = predicted_labels.numpy()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Background (Class 0)', 'Signal (Class 1)']))

    np.savez(
        save_path,
        discriminant_scores = discriminant_scores.numpy(),
        Lumi_weights = lumi_test_tensor.numpy(),
        y_true = y_true,
        y_pred = y_pred
    )


if __name__ == '__main__':

    date = datetime.today().strftime('%Y%m%d')
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_path', type = str, default = '../data/processed/data_tensors.pt', required= False)
    parser.add_argument('--model_path',type=str,required=False,default = 'ModelsDNN/dnn_model.pth')
    parser.add_argument('--numpy_save_filename', type = str,required=False, default = f'../data/dnn_discriminant_scores_{date}.npz' )
    args = parser.parse_args()
    main(tensor_path = args.tensor_path, model_path= args.model_path, save_path = args.numpy_save_filename)