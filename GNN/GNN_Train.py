from GNN import GCN
import torch
from preprocess_graph_data import CPDataSet
from torch_geometric.data import DataLoader


def main():
    dataset = CPDataSet(root = '../graphdata/CP_Studies_llqq_graphs')

    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    num_graphs = int(len(dataset))
    split_idx = int(0.8 * len(dataset))

    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    print(f"Training set size: {len(train_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")


    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    input_size = dataset.num_node_features
    hidden_dim = 16
    learning_rate = 0.01

    model = GCN(input_size, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 5e-4)
    criterion = torch.nn.BCELoss()

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            out = model(batch.x,batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.float().unsqueeze(1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")

    
if __name__ == "__main__":
    main()


    