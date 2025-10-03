from preprocess_data import X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, weights
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

input_dim = X_train_tensor.shape[1]
hidden_dim = 64
output_dim = 1 #Binary Classifcation

print("\nDefining the DNN model...")
class DNN(nn.Module):
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
    
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
torch.save(model.state_dict(), 'dnn_model.pth')

print("\nEvaluating the model on test data")
model.eval()
with torch.no_grad():
    probabilities_p_plus = model(X_test_tensor)

    p_plus = probabilities_p_plus.squeeze() #make 1d
    p_minus = 1 - p_plus
    discriminant_scores = p_plus - p_minus
    predicted_labels = (probabilities_p_plus > 0.5).long().squeeze()

    y_true = y_test_tensor.numpy()
    y_pred = predicted_labels.numpy()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Background (Class 0)', 'Signal (Class 1)']))


print("\n--- Final Output Inspection ---")
print("Shape of probabilities_x p_plus tensor:", probabilities_p_plus.shape)
print("Shape of discriminant_scores tensor:", discriminant_scores.shape)

print("\nExample outputs for the first 5 test events:")
for i in range(5):
    print(f"Event {i}:")
    print(f" Model Output   : [p(+): {p_plus[i]:.4f}")
    print(f"  Discriminant (p(+) - p(-)): {discriminant_scores[i]:.4f}")
    print(f"  Predicted Label: {predicted_labels[i].item()}, True Label: {y_test_tensor[i].item()}")
    print("-" * 20)