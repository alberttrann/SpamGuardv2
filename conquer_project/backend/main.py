from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import asyncio
import json

from classifier import SpamGuardClassifier
import database
import llm_generator
from train_nb import retrain_and_save

app = FastAPI(title="SpamGuard AI API", version="1.1.0")

# --- App Startup: Initialize DB and load classifier ---
@app.on_event("startup")
def startup_event():
    database.init_db()
    # Using a global dictionary to hold the classifier instance
    # This allows us to replace the instance after retraining
    app.state.classifier = SpamGuardClassifier()

# --- Pydantic Models ---
class Message(BaseModel):
    text: str
class Feedback(BaseModel):
    message: str
    correct_label: str
# --- Pydantic Models (Updated) ---
class LLMRequest(BaseModel):
    provider: str # 'ollama' or 'openrouter'
    model: str
    api_key: str | None = None
    label_to_generate: str | None = None # 'spam', 'ham', or None for random

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the SpamGuard AI API"}

@app.post("/classify")
def classify_message(message: Message):
    result = app.state.classifier.classify(message.text)
    return result

@app.post("/feedback")
def receive_feedback(feedback: Feedback):
    database.add_feedback(message=feedback.message, label=feedback.correct_label)
    return {"status": "success", "message": "Feedback received. Thank you!"}

@app.get("/analytics")
def get_analytics():
    return database.get_analytics()

@app.post("/retrain")
async def trigger_retraining():
    try:
        print("--- API: Retraining process initiated. ---")
        # 1. Enrich dataset with new feedback
        new_records_count = database.enrich_main_dataset()
        if new_records_count == 0:
            return {"status": "skipped", "message": "No new feedback data to train on."}

        # 2. Retrain Naive Bayes model and save artifacts
        retrain_and_save()

        # 3. Reload the classifier instance in the running app
        app.state.classifier.reload()
        
        print("--- API: Retraining process completed successfully. ---")
        return {"status": "success", "message": f"Model retrained with {new_records_count} new records."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- API Endpoints (Updated) ---
@app.post("/generate_data")
async def generate_data_stream(req: LLMRequest):
    async def event_stream():
        if req.provider == 'ollama':
            generator = llm_generator.generate_with_ollama(
                model=req.model, 
                label_to_generate=req.label_to_generate
            )
        elif req.provider == 'openrouter':
            if not req.api_key:
                raise HTTPException(status_code=400, detail="API key is required for OpenRouter.")
            generator = llm_generator.generate_with_openrouter(
                model=req.model, 
                api_key=req.api_key, 
                label_to_generate=req.label_to_generate
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid provider specified.")

        async for status in generator:
            if status.startswith("Generated:"):
                try:
                    json_str = status.replace("Generated: ", "").replace("'", '"')
                    data = json.loads(json_str)
                    database.add_feedback(data['message'], data['label'], source='llm')
                except Exception as e:
                    print(f"Error saving generated data: {e}")
            
            yield f"data: {status}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")