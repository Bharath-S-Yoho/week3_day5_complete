"""Simple FastAPI endpoints for support ticket predictions."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

app = FastAPI(title="Customer Support Ticket Classifier", version="1.0")
MODEL_DIR = Path(__file__).resolve().parents[1] / "task2_model" / "models"

class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Support ticket text")

def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise RuntimeError("Models not found. Run: python task2_model/train_model.py")
    return joblib.load(path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(ticket: TicketRequest):
    try:
        category_model = load_model("category_model.joblib")
        priority_model = load_model("priority_model.joblib")
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    text = ticket.text.strip().lower()
    return {
        "text": ticket.text,
        "category": str(category_model.predict([text])[0]),
        "priority": str(priority_model.predict([text])[0])
    }
