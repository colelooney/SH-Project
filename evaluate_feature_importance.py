from preprocess_data import X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, weights
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import h5py
import pandas as pd
from DNN_SignalvsBackground import DNN


path = f'ModelsDNN/dnn_model.pth'
model = DNN(input_size = X_train_tensor.shape[1])
model.load_state_dict(torch.load(path))

mi = mutual_info_classif(X_train_tensor.numpy(), y_train_tensor.numpy(), discrete_features = 'auto')


relative_path = './data/new_Input_CP_Studies_llqq_LinearTerm_29_September2025.h5'
with h5py.File(relative_path) as f:
    df = pd.DataFrame(f['LargeRJet']['1d'][:])

feature_names = df.drop(columns=['Lumi_weight']).columns.tolist()


feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': mi.flatten()})

feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)