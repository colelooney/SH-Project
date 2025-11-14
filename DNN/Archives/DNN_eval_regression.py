"""
Cole Looney — Updated for Interference Regression (Nov 2025)

DNN_eval_regression.py

Evaluate trained DNN model on test data (regression target = lumi_weight)
Compute regression metrics (MSE, correlation, sign accuracy)
and plot CP-odd style histograms.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import argparse
from DNN_models import DNN
import os

def main(tensor_path, model_path, save_path):
    # --- Load preprocessed data ---
    data_dict = torch.load(tensor_path)
    X_test_tensor = data_dict['X_test']
    lumi_test_tensor = data_dict['lumi_test']  # target = interference weight

    input_size = X_test_tensor.shape[1]

    # --- Load trained model ---
    model = DNN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        predictions = model(X_test_tensor).squeeze()

    # --- Convert to numpy for analysis ---
    lumi_mean = lumi_test_tensor.mean()
    lumi_std = lumi_test_tensor.std()
    lumi_test_tensor = (lumi_test_tensor - lumi_mean) / lumi_std

    y_true = (lumi_test_tensor.numpy() - lumi_mean.numpy()) / lumi_std.numpy()
    y_pred = predictions.numpy()

    # --- Regression metrics ---
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    sign_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    print("\n--- Regression Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"Pearson Correlation: {corr:.6f}")
    print(f"Sign Accuracy: {sign_acc:.3f}")

    # --- Save results ---
    np.savez(
        save_path,
        y_true=y_true,
        y_pred=y_pred,
        lumi_weights=y_true  # same as target, for plotting later
    )

    # --- Optional: CP-odd Discriminant Plot ---
    plt.figure(figsize=(8,6))
    plt.hist(y_pred, bins=75, weights=y_true, histtype='step', linewidth=2, color='blue')
    plt.title('Predicted CP-Odd Discriminant (Weighted by Lumi Weight)')
    plt.xlabel('Predicted Interference Term (arb. units)')
    plt.ylabel('Weighted Event Density')
    plt.grid(True, alpha=0.3)
    plt.savefig('../plots/DNN_CPodd_Discriminant.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_path', type=str, default='../data/processed/data_tensors_even.pt')
    parser.add_argument('--model_path', type=str, default='ModelsDNN/dnn_model_mse.pth')
    parser.add_argument('--save_path', type=str, default='../data/dnn_regression_outputs.npz')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    main(args.tensor_path, args.model_path, args.save_path)
