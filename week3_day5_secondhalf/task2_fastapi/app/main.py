from pathlib import Path
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models only once during startup.
    for target in ("category", "priority"):
        path = MODEL_DIR / f"{target}_model.joblib"
        if path.exists():
            models[target] = joblib.load(path)
        else:
            models[target] = None
    yield
    models.clear()

app = FastAPI(title="Customer Support Ticket Classifier - Part 2", lifespan=lifespan)

class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Ticket text must not be empty")

class BatchRequest(BaseModel):
    tickets: List[TicketRequest]

def fallback_predict(text):
    """Small demo fallback used only if Part 1 joblib files were not copied."""
    lower = text.lower()
    if any(word in lower for word in ["charged", "payment", "invoice", "refund", "bill"]):
        category = "Billing"
    elif any(word in lower for word in ["delivery", "package", "shipment", "parcel", "tracking"]):
        category = "Delivery"
    elif any(word in lower for word in ["password", "account", "profile", "login", "log in"]):
        category = "Account"
    else:
        category = "Technical"
    priority = "High" if any(word in lower for word in ["urgent", "immediately", "as soon as possible"]) else "Medium"
    return category, priority, 0.70

def predict_one(text):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty")
    category_model = models.get("category")
    priority_model = models.get("priority")
    if category_model is None or priority_model is None:
        category, priority, confidence = fallback_predict(text)
        return {"category": category, "priority": priority, "confidence": confidence}
    category = str(category_model.predict([text.lower()])[0])
    priority = str(priority_model.predict([text.lower()])[0])
    category_conf = float(max(category_model.predict_proba([text.lower()])[0]))
    priority_conf = float(max(priority_model.predict_proba([text.lower()])[0]))
    return {"category": category, "priority": priority, "confidence": round((category_conf + priority_conf) / 2, 3)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(ticket: TicketRequest):
    return predict_one(ticket.text)

@app.post("/predict-batch")
def predict_batch(batch: BatchRequest):
    return {"predictions": [predict_one(ticket.text) for ticket in batch.tickets]}
