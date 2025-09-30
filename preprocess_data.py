
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

relative_path = './data/new_Input_CP_Studies_llqq_LinearTerm_29_September2025.h5'
with h5py.File(relative_path) as f:
    df = pd.DataFrame(f['LargeRJet']['1d'][:])

X = df.drop(columns=['Lumi_weight'])
y = df['Lumi_weight']

y[y>0] = 1
y[y<0] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().transform(X_test)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
weights = torch.tensor([1.0,1.0]) #for output classes 0 and 1, approximately same number of events