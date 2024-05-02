import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

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

def load_autoencoder_model(model_path):
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))
    return model

def predict_with_autoencoder(embedding, weights, autoencoder_model_path="autoencoder_model.pth", data_path="data/data.csv"):
    autoencoder_model = load_autoencoder_model(autoencoder_model_path)
    autoencoder_model.eval()

    data = pd.read_csv(data_path)

    input_data = convert_embedding(embedding)
    input_data = standardize_data(input_data, weights)
    input_data = torch.tensor(input_data, dtype=torch.float)

    num_nodes = input_data.size(0)
    edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()

    # Predict
    with torch.no_grad():
        output = autoencoder_model(Data(x=input_data, edge_index=edge_index))
        _, predicted_index = torch.max(output, 1)
        predicted_song = data.iloc[predicted_index.item()]
        predicted_song_name = predicted_song["name"]
        predicted_artist = predicted_song["artists"]
        predicted_year = predicted_song["year"]
        return predicted_song_name, predicted_artist, predicted_year    

def convert_embedding(embedding):
    column_order = ['valence', 'key', 'tempo', 'acousticness', 'danceability', 'energy', 'explicit', 'instrumentalness', 'liveness', 'speechiness', 'loudness', 'year']
    df = pd.DataFrame.from_dict(embedding, orient='index').T
    df = df[column_order]
    return df

def standardize_data(data, weights=None):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if weights is not None:
        data *= weights
    return data