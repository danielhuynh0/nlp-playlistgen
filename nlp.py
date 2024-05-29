import pathlib
import textwrap
import os
import json
import kmeans
import ast

import google.generativeai as genai
import gnn
import autoencoder

prompt_instructions = 'There are embeddings for songs that follow the following structure: \"valence, a measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry), key of the song ranged from 0 to 11 with label key, tempo of the song with label tempo, acousticness, measured from 0 to 1, danceability, describing how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity where a value of 0.0 is least danceable and 1.0 is most danceable, energy, measured from 0 to 1, explicit, a discrete variable that is either 0 or 1 depending on if the song is explicit or not, instrumentalness, measured from 0 to 1, liveness, measured from 0 to 1, which determines the presence of an audience or not, speechiness, measured from 0 to 1, and loudness, or the overall loudness of a track in decibels (dB), averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db. Finally, give me a year that the song will likely have been released.\" I will give a description for an emotion I want elicited in a song, and I want you to use the embedding structure I provided to generate an embedding for the song I describe. For continuous properties that are from 0 to 1, use up to 5 decimal points for greater precision. Generate the embedding in a JSON format. For the song embedding, it must be valid json format with keys: valence, key, tempo, acousticness, danceability, energy, explicit, instrumentalness, liveness, speechiness, loudness, and year. Make this embedding object have the key \"embedding\". In addition, generate a list of weights that rate each embedding category based on its importance from the description. Make this list a part of the JSON object that is returned, with a key called \"weights\" and as a list object without the corresponding keys (only have the values, in the same order as the json object for the embedding). If the description asks for a modern song for example, make sure to add more importance to the year. Being able to identify the correct weights based on what is stated in the description is extemely important. Make the embedding and the weights be nested inside of the final json object returned. Return nothing but the JSON object, no other words, do not even put "JSON" at the top. The description is: \"'
from dotenv import load_dotenv

def get_embedding_from_description(description):
    load_dotenv()
    key = os.environ.get('GOOGLE_API_KEY')
    model = genai.GenerativeModel('gemini-pro')
    genai.configure(api_key=key)
    response = model.generate_content(prompt_instructions+description+'\"')
    response = response.text
    json_content = response.replace('```json', '').replace('```', '').strip()
    print(json_content)
    data = json.loads(json_content)
    return data

def run(description, number_of_songs):
    embedding = get_embedding_from_description(description)
    song_embedding = embedding['embedding']
    for key in song_embedding:
        if(song_embedding[key] == None):
            song_embedding[key] = 0
    weights = embedding['weights']
    prediction = kmeans.predict(song_embedding, weights, number_of_songs)
    data_fixed_quotes = [[item.replace('"', "'") for item in sublist] for sublist in prediction]
    result = [tuple(item) for item in data_fixed_quotes]
    return result

def main():
    description = input("Enter a description of the type of songs you like: ")
    number_of_songs = int(input("Enter the number of songs you want to generate: "))
    embedding = get_embedding_from_description(description)
    song_embedding = embedding['embedding']
    weights = embedding['weights']
    prediction = kmeans.predict(song_embedding, weights, number_of_songs)
    data_fixed_quotes = [[item.replace('"', "'") for item in sublist] for sublist in prediction]

    result = [tuple(item) for item in data_fixed_quotes]
    print("Here are the songs that match your description as a result of K Nearest Neighbors: ")
    print(result)
    # print("Here is the song that most matched your description from the graph neural network (GNN): ")
    # gnn_result = gnn.predict_with_gnn(song_embedding, weights)
    # print(gnn_result)
    # print("Here is the song that most matched your description from the autoencoder: ")
    # autoencoder_result = autoencoder.predict_with_autoencoder(song_embedding, weights)
    # print(autoencoder_result)

if __name__ == '__main__':
    main()