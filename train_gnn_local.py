import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import json

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(12, 16)
        self.conv2 = GCNConv(16, 8)
        self.fc1 = nn.Linear(8, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.sigmoid(self.fc1(x))
        return x

# Load data
def get_data():
    data_file = "data/data.csv"
    data = pd.read_csv(data_file)
    return data

# Prepare data for training and testing
def get_XY(data, weights=None):
    features = ['valence', 'key', 'tempo', 'acousticness', 'danceability', 
                'energy', 'explicit', 'instrumentalness', 'liveness', 
                'speechiness', 'loudness', 'year']
    X = data[features]
    Y = data['id'] if 'id' in data.columns else None
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Apply weights
    if weights is not None:
        for i, weight in enumerate(weights):
            X[:, i] *= weight
    
    return X, Y

# Train the GNN model
def train_gnn(data, weights, epochs=100):
    model = Net()

    X, Y = get_XY(data, weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(Data(x=torch.tensor(X, dtype=torch.float), edge_index=torch.tensor([[i, i] for i in range(X.shape[0])], dtype=torch.long).t().contiguous()))
        loss = criterion(out, torch.tensor([[1.]] * X.shape[0], dtype=torch.float))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

    # Save model
    torch.save(model.state_dict(), 'gnn_model.pth')



# Convert JSON embedding to compatible format
def convert_json_to_embedding(json_embedding):
    embedding_data = json.loads(json_embedding)
    embedding = embedding_data["embedding"]
    weights = embedding_data["weights"]
    return embedding, weights

# Example usage:
json_embedding = '''
{
    "embedding": {
        "valence": 0.9,
        "key": 7,
        "tempo": 120.0,
        "acousticness": 0.2,
        "danceability": 0.8,
        "energy": 0.7,
        "explicit": 0,
        "instrumentalness": 0.1,
        "liveness": 0.5,
        "speechiness": 0.2,
        "loudness": -10.0,
        "year": 2020
    },
    "weights": [0.8, 0.3, 0.7, 0.6, 0.9, 0.8, 0.1, 0.4, 0.5, 0.4, 0.9, 1.0]
}
'''

'''
We ended up with the following results:

Epoch: 0, Loss: 0.6387896537780762
Epoch: 10, Loss: 0.35116046667099
Epoch: 20, Loss: 0.1375829428434372
Epoch: 30, Loss: 0.027752600610256195
Epoch: 40, Loss: 0.005151386838406324
Epoch: 50, Loss: 0.0015584159409627318
Epoch: 60, Loss: 0.0007890438428148627
Epoch: 70, Loss: 0.0005430469755083323
Epoch: 80, Loss: 0.0004343124164734036
Epoch: 90, Loss: 0.00037211639573797584


'''

