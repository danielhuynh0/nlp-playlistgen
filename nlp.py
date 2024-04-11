import pathlib
import textwrap
import os

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

prompt_instructions = 'There are embeddings for songs that follow the following structure: \"valence, a measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry), acousticness, measured from 0 to 1, danceability, describing how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity where a value of 0.0 is least danceable and 1.0 is most danceable, energy, measured from 0 to 1, explicit, a discrete variable that is either 0 or 1 depending on if the song is explicit or not, instrumentalness, measured from 0 to 1, liveness, measured from 0 to 1, which determines the presence of an audience or not, speechiness, measured from 0 to 1, and loudness, or the overall loudness of a track in decibels (dB), averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.\" I will give a description for an emotion I want elicited in a song, and I want you to use the embedding structure I provided to generate an embedding for the song I describe. Generate the embedding in a JSON format. The description is: \"'
model = genai.GenerativeModel('gemini-pro')
key = os.environ.get('GOOGLE_API_KEY')

def get_embedding_from_description(description):
    response = model.generate_content(prompt_instructions+description+'\"', key=key)
    return response
