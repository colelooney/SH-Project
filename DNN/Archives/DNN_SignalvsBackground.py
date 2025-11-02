from preprocess_large_data import X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, lumi_test, lumi_train
from preprocess_large_data import quad_lumi, quad_tensor
import torch
# from preprocess_data import X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, lumi_test
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

input_dim = X_train_tensor.shape[1]
hidden_dim = 128
output_dim = 1 #Binary Classifcation

class DNN(nn.Module):
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim,output_dim)
            # nn.Sigmoid()
        )

    def forward(self, x):
        # return self.layers(x)
        return self.layers(x) #logits shape (N,)

def main():
    input_size = X_train_tensor.shape[1]
    learning_rate = 0.0001
    batch_size = 128
    num_epochs = 20

    epochs = []
    losses = []

    model = DNN(input_size)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
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
        losses.append(loss.item())
        epochs.append(epoch)

    #Save the model
    torch.save(model.state_dict(), '../ModelsDNN/dnn_model_even.pth')

    np.savez(
        '../../data/10312025_even_loss.npz',
        losses = np.array(losses),
        epochs=  np.array(epochs)
    )

    # model.eval()
    # with torch.no_grad():
    #     out = model(X_train_tensor)    # shape (N,1)
    #     out = torch.sigmoid(out)
    #     print('train outputs mean,min,max:', out.mean().item(), out.min().item(), out.max().item())
    #     out_test = model(X_test_tensor)
    #     out_test = torch.sigmoid(out_test)
    #     print('test outputs mean,min,max:', out_test.mean().item(), out_test.min().item(), out_test.max().item())

    # model.train()
    # features, labels = next(iter(train_loader))
    # optimizer.zero_grad()
    # out = model(features)
    # loss = criterion(out, labels.unsqueeze(1))
    # loss.backward()
    # total_grad_norm = 0.0
    # for p in model.parameters():
    #     if p.grad is not None:
    #         total_grad_norm += p.grad.data.norm(2).item()**2
    # print('grad norm sqrt:', total_grad_norm**0.5)

    # print("\nEvaluating the model on validation data")
    # model.eval()
    # with torch.no_grad():
    #     out = model(X_dev_tensor)

    #     probabilities_p_plus = torch.sigmoid(out)

    #     p_plus = probabilities_p_plus.squeeze() #make 1d
    #     p_minus = 1 - p_plus
    #     discriminant_scores = p_plus - p_minus
    #     predicted_labels = (probabilities_p_plus > 0.5).long().squeeze()

    #     y_true = y_dev_tensor.numpy()
    #     y_pred = predicted_labels.numpy()

    #     print('roc ', roc_auc_score(y_true,p_plus))

    #     print("\nClassification Report:")
    #     print(classification_report(y_true, y_pred, target_names=['Background (Class 0)', 'Signal (Class 1)']))


    # print("\n--- Final Output Inspection ---")
    # print("Shape of probabilities_x p_plus tensor:", probabilities_p_plus.shape)
    # print("Shape of discriminant_scores tensor:", discriminant_scores.shape)

    # print("\nExample outputs for the first 5 test events:")
    # for i in range(5):
    #     print(f"Event {i}:")
    #     print(f" Model Output   : [p(+): {p_plus[i]:.4f}")
    #     print(f"  Discriminant (p(+) - p(-)): {discriminant_scores[i]:.4f}")
    #     print(f"  Predicted Label: {predicted_labels[i].item()}, True Label: {y_dev_tensor[i].item()}")
    #     print("-" * 20)


    # #save discriminant scores and lumi weights for plotting
    # np.savez(
    #     f'../../data/dnn_discriminant_scores_and_lumi_weights_10302025_even_val.npz',
    #     discriminant_scores = discriminant_scores.numpy(),
    #     Lumi_weights = np.array(lumi_dev),
    #     y_true = y_true,
    #     y_pred = y_pred
    # )

    print("\nEvaluating the model on train data")
    model.eval()
    with torch.no_grad():
        out = model(X_train_tensor)

        probabilities_p_plus = torch.sigmoid(out)

        p_plus = probabilities_p_plus.squeeze() #make 1d
        p_minus = 1 - p_plus
        discriminant_scores = p_plus - p_minus
        predicted_labels = (probabilities_p_plus > 0.5).long().squeeze()

        y_true = y_train_tensor.numpy()
        y_pred = predicted_labels.numpy()

        print('roc ', roc_auc_score(y_true,p_plus))

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
        print(f"  Predicted Label: {predicted_labels[i].item()}, True Label: {y_train_tensor[i].item()}")
        print("-" * 20)


    #save discriminant scores and lumi weights for plotting
    np.savez(
        f'../../data/dnn_discriminant_scores_and_lumi_weights_10302025_even_train.npz',
        discriminant_scores = discriminant_scores.numpy(),
        Lumi_weights = np.array(lumi_train),
        y_true = y_true,
        y_pred = y_pred
    )

    print("\nEvaluating the model on test data")
    model.eval()
    with torch.no_grad():
        out = model(X_test_tensor)

        probabilities_p_plus = torch.sigmoid(out)

        p_plus = probabilities_p_plus.squeeze() #make 1d
        p_minus = 1 - p_plus
        discriminant_scores = p_plus - p_minus
        predicted_labels = (probabilities_p_plus > 0.5).long().squeeze()

        y_true = y_test_tensor.numpy()
        y_pred = predicted_labels.numpy()

        print('roc ', roc_auc_score(y_true,p_plus))

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


    #save discriminant scores and lumi weights for plotting
    np.savez(
        f'../../data/dnn_discriminant_scores_and_lumi_weights_10302025_eventest.npz',
        discriminant_scores = discriminant_scores.numpy(),
        Lumi_weights = np.array(lumi_test),
        y_true = y_true,
        y_pred = y_pred
    )

    print("\nGetting predictions for Quadratic Term data")
    model.eval()
    with torch.no_grad():
        out = model(quad_tensor)

        probabilities_p_plus = torch.sigmoid(out)

        p_plus = probabilities_p_plus.squeeze() #make 1d
        p_minus = 1 - p_plus
        discriminant_scores = p_plus - p_minus

    #save discriminant scores and lumi weights for plotting
    np.savez(
        f'../../data/dnn_discriminant_scores_and_lumi_weights_10302025_even_quad.npz',
        discriminant_scores = discriminant_scores.numpy(),
        Lumi_weights = np.array(quad_lumi)
    )
if __name__ == "__main__":
    main()