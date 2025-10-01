from preprocess_data import X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, weights
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

input_dim = X_train_tensor.shape[1]
hidden_dim = 64
output_dim = 2

print("\nStep 2: Defining the DNN model...")
class DNN(nn.Module):
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)
    
input_size = X_train_tensor.shape[1]
learning_rate = 0.001
batch_size = 256
num_epochs = 20

model = DNN(input_size)
criterion = nn.CrossEntropyLoss(weight = weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

print(f"\nStarting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    for i, (features, labels) in enumerate(train_loader):
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\nEvaluating the model on test data")
model.eval()
with torch.no_grad():
    raw_predictions = model(X_test_tensor)

    probabilities = torch.nn.functional.softmax(raw_predictions, dim=1)
    p_minus = probabilities[:, 0]
    p_plus = probabilities[:, 1]
    discriminant_scores = p_plus - p_minus
    _, predicted_labels = torch.max(raw_predictions, 1)
    correct = (predicted_labels == y_test_tensor).sum().item()
    total = y_test_tensor.size(0)
    accuracy = 100 * correct / total
    print(f'\nAccuracy on the test set: {accuracy:.2f} %')


print("\n--- Final Output Inspection ---")
print("Shape of raw_predictions tensor:", raw_predictions.shape)
print("Shape of probabilities tensor:", probabilities.shape)
print("Shape of discriminant_scores tensor:", discriminant_scores.shape)

print("\nExample outputs for the first 5 test events:")
for i in range(5):
    print(f"Event {i}:")
    print(f"  Raw Logits   : [Class 0: {raw_predictions[i, 0]:.4f}, Class 1: {raw_predictions[i, 1]:.4f}]")
    print(f"  Probabilities: [p(-): {p_minus[i]:.4f}, p(+): {p_plus[i]:.4f}]")
    print(f"  Discriminant (p(+) - p(-)): {discriminant_scores[i]:.4f}")
    print("-" * 20)