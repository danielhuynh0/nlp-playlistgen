import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 

def get_data():
    data_file = "data/data.csv"
    data = pd.read_csv(data_file)
    return data

from sklearn.preprocessing import StandardScaler

def get_XY(data, weights=None):
    features = ['valence', 'key', 'tempo', 'acousticness', 'danceability', 'energy', 'explicit', 'instrumentalness', 'liveness', 'speechiness', 'loudness', 'year']
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

def get_model(X, Y, n_neighbors=7):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, Y_train)
    return knn

def get_predictions(model, data, X, n_neighbors):
    distances, indices = model.kneighbors(X, n_neighbors=n_neighbors)
    predictions = []
    for index in indices[0]:
        predictions.append([data.loc[index, 'artists'], data.loc[index, 'name'], data.loc[index, 'release_date']])
    return predictions

def predict(embedding, weights, n_neighbors=7):
    data = get_data()
    X, Y = get_XY(data, weights)
    model = get_model(X, Y, n_neighbors=n_neighbors)
    my_embedding = convert_embedding(embedding)
    predictions = get_predictions(model, data, my_embedding, n_neighbors=n_neighbors)
    return predictions

def convert_embedding(embedding):
    print(embedding)
    column_order = ['valence', 'key', 'tempo', 'acousticness', 'danceability', 'energy', 'explicit', 'instrumentalness', 'liveness', 'speechiness', 'loudness', 'year']
    df = pd.DataFrame.from_dict(embedding, orient='index').T
    df = df[column_order]
    return df