
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
# relative_path = '../data/new_Input_CP_Studies_llqq_LinearTerm_13th_October2025.h5'
relative_path = '../../data/s2286706/new_Input_CP_Studies_llqq_LinearTerm_20th_October2025.h5'
with h5py.File(relative_path) as f:
    df = pd.DataFrame(f['LargeRJet']['1d'][:])

train_dev = df[df['EventNumber'] % 2 ==0] # train on even event numbers
split_idx = int(0.8*len(train_dev))
# train = train_dev[:split_idx]
# dev = train_dev[split_idx:]
train = train_dev
test = df[df['EventNumber'] % 2 == 1] # test on odd event numbers

print(train)

lumi_train =  train['Lumi_weight'].copy()
# lumi_dev = dev['Lumi_weight'].copy()
lumi_test = test['Lumi_weight'].copy()

y_train = train['Lumi_weight'].copy()
X_train = train.drop(columns=['Lumi_weight','EventNumber','FJ_flavour'])

# y_dev = dev['Lumi_weight'].copy()
# X_dev = dev.drop(columns=['Lumi_weight','EventNumber','FJ_flavour'])

y_test = test['Lumi_weight'].copy()
X_test = test.drop(columns=['Lumi_weight','EventNumber','FJ_flavour'])

y_train[y_train>0] = 1
y_train[y_train<0] = 0

# y_dev[y_dev>0] = 1
# y_dev[y_dev<0] = 0

y_test[y_test>0] = 1
y_test[y_test<0] = 0

y_train = np.array(y_train)
# y_dev = np.array(y_dev)
y_test = np.array(y_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# X_dev_tensor = torch.tensor(X_dev,dtype=torch.float32)

# y_dev_tensor = torch.tensor(y_dev,dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

lumi_train_tensor = torch.tensor(lumi_train.to_numpy(),dtype=torch.float32)
# lumi_dev_tensor = torch.tensor(lumi_dev.to_numpy(),dtype=torch.float32)
lumi_test_tensor = torch.tensor(lumi_test.to_numpy(), dtype=torch.float32)

quadratic_path = '../../data/s2286706/new_Input_CP_Studies_llqq_QuadraticTerm_20th_October2025.h5'
with h5py.File(quadratic_path) as f:
    df = pd.DataFrame(f['LargeRJet']['1d'][:])

quad_test = df
quad_lumi = quad_test['Lumi_weight'].copy()
quad_train = quad_test.drop(columns= ['Lumi_weight','EventNumber','FJ_flavour'])

quad_train = scaler.transform(quad_train)

quad_tensor = torch.tensor(quad_train,dtype=torch.float32)
