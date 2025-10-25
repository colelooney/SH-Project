from preprocess_large_data import X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, lumi_test_tensor
import torch
# from preprocess_data import X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, lumi_test
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from DNN_models.py import DNN

def main():
    input_size = X_train_tensor.shape[1]
    learning_rate = 0.01
    batch_size = 256
    num_epochs = 20

    model = DNN(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.float())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    print(model)

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        for i, (features, labels) in enumerate(train_loader):
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    #Save the model
    torch.save(model.state_dict(), 'ModelsDNN/dnn_model.pth')