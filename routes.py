from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile
import nlp
import kmeans

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    description: str = Field(..., example="a happy sunny day at the beach")
    number_of_songs: int = Field(..., example=5)

@app.post("/predict/")
async def predict(item: Item):
    description = item.description
    number_of_songs = item.number_of_songs
    result = nlp.run(description, number_of_songs)
    return {"result": result}