import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from train_gnn_local import Net
from torch_geometric.data import Data

def load_gnn_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    return model

def predict_with_gnn(embedding, weights, gnn_model_path="gnn_model.pth", data_path="data/data.csv"):
    # Load GNN model
    gnn_model = load_gnn_model(gnn_model_path)
    gnn_model.eval()

    # Load song dataset
    data = pd.read_csv(data_path)

    # Prepare data
    input_data = convert_embedding(embedding)
    input_data = standardize_data(input_data, weights)
    input_data = torch.tensor(input_data, dtype=torch.float)

    # Prepare edge index
    num_nodes = input_data.size(0)
    edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()

    # Predict
    with torch.no_grad():
        output = gnn_model(Data(x=input_data, edge_index=edge_index))
        _, predicted_index = torch.max(output, 1)
        predicted_song_name = data.iloc[predicted_index.item()]["name"]
        return predicted_song_name
    

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
