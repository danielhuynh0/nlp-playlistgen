import pathlib
import textwrap
import os

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

prompt_instructions = ""
model = genai.GenerativeModel('gemini-pro')
key = os.environ.get('GOOGLE_API_KEY')

def get_embedding_from_description(description):
    response = model.generate_content(prompt_instructions+description, key=key)
    return response
