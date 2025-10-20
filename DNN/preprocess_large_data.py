
"""
Cole Looney

Data Preprocessing Script
"""

import h5py
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# relative_path = './data/new_Input_CP_Studies_llqq_LinearTerm_29_September2025.h5' #Path to first data file
relative_path = '../data/new_Input_CP_Studies_llqq_LinearTerm_13th_October2025.h5'
with h5py.File(relative_path) as f:
    df = pd.DataFrame(f['LargeRJet']['1d'][:])

train = df['event_number'] % 2 == 0 # train on even event numbers
test = df['event_number'] % 2 == 1 # test on odd event numbers

X_train = train.drop(columns=['Lumi_weight','event_number'])
y_train = train['Lumi_weight'].copy()

X_test = test.drop(columns=['Lumi_weight','event_number'])
y_test = test['Lumi_weight'].copy()
lumi_train =  train['Lumi_weight'].copy()
lumi_test = test['Lumi_weight'].copy()

y_train[y_train>0] = 1
y_train[y_train<0] = 0

y_test[y_test>0] = 1
y_test[y_test<0] = 0

y_train = np.array(y_train)
y_test = np.array(y_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
lumi_test_tensor = torch.tensor(lumi_test.to_numpy(), dtype=torch.float32)