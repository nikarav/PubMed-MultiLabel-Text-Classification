from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.model import Model

app = FastAPI()

model = Model()


class PredictionRequest(BaseModel):
    texts: str


class PredictionResponse(BaseModel):
    prediction: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not request.texts:
        raise HTTPException(
            status_code=400, detail="Input text list is empty"
        )  # no text provided

    try:
        prediction = model.predict(request.texts)
        return PredictionResponse(prediction=prediction)  # return text answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "PubMed multi-label classification API."}
