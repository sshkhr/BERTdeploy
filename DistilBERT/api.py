from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .classifier import BERTClassifier, get_bert

app = FastAPI()

class ClassificationRequest(BaseModel):
	text: str

class ClassificationResponse(BaseModel):
	probabilities: Dict[str, float]
	sentiment: str
	confidence: float


@app.post("/classify", response_model = ClassificationResponse)
def classify(request: ClassificationRequest, model: BERTClassifier = Depends(get_bert)):
	sentiment, confidence, probabilities = model.predict(request.text)
	return ClassificationResponse(
		sentiment = sentiment, 
		confidence = confidence, 
		probabilities = probabilities
	)