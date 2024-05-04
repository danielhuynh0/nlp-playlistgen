import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import json

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(12, 16)
        self.conv2 = GCNConv(16, 8)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = GCNConv(8, 16)
        self.conv2 = GCNConv(16, 12)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.sigmoid(self.conv2(x, edge_index))
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        return x

def get_data():
    data_file = "data/data.csv"
    data = pd.read_csv(data_file)
    return data

def get_XY(data, weights=None):
    features = ['valence', 'key', 'tempo', 'acousticness', 'danceability', 
                'energy', 'explicit', 'instrumentalness', 'liveness', 
                'speechiness', 'loudness', 'year']
    X = data[features]
    Y = data['id'] if 'id' in data.columns else None

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if weights is not None:
        for i, weight in enumerate(weights):
            X[:, i] *= weight

    return X, Y

from sklearn.neighbors import NearestNeighbors

def get_edge_index(X, k=5):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    edge_index = []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):  # Ignore the first neighbor because it's the node itself
            edge_index.append([i, indices[i, j]])

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def train_autoencoder(data, weights, epochs=100):
    model = Autoencoder()

    X, _ = get_XY(data, weights)
    X = torch.tensor(X, dtype=torch.float)
    edge_index = get_edge_index(X.numpy())  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(Data(x=X, edge_index=edge_index))
        loss = criterion(out, X)  # Compare the output to the original input
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'autoencoder_model.pth')

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

data = get_data()
embedding, weights = convert_json_to_embedding(json_embedding)
train_autoencoder(data, weights)