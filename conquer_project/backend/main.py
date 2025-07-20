from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import asyncio
import json

from classifier import SpamGuardClassifier
import database
import llm_generator
from train_nb import retrain_and_save

app = FastAPI(title="SpamGuard AI API", version="1.3.0")

# --- (No changes to startup or other endpoints) ---
@app.on_event("startup")
def startup_event():
    database.init_db()
    app.state.classifier = SpamGuardClassifier()

class Message(BaseModel):
    text: str
class Feedback(BaseModel):
    message: str
    correct_label: str
class LLMRequest(BaseModel):
    provider: str
    model: str
    api_key: str | None = None
    label_to_generate: str | None = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the SpamGuard AI API"}

# ... (classify, feedback, analytics, retrain endpoints are unchanged) ...
@app.post("/classify")
def classify_message(message: Message):
    return app.state.classifier.classify(message.text)

@app.post("/feedback")
def receive_feedback(feedback: Feedback):
    database.add_feedback(message=feedback.message, label=feedback.correct_label)
    return {"status": "success", "message": "Feedback received."}

@app.get("/analytics")
def get_analytics():
    return database.get_analytics()

@app.post("/retrain")
async def trigger_retraining():
    new_records_count = database.enrich_main_dataset()
    if new_records_count == 0:
        return {"status": "skipped", "message": "No new feedback data to train on."}
    retrain_and_save()
    app.state.classifier.reload()
    return {"status": "success", "message": f"Model retrained with {new_records_count} new records."}


# --- CHANGE: Adjusted Data Generation Endpoint to handle new generator output ---
@app.post("/generate_data")
async def generate_data_stream(req: LLMRequest, raw_request: Request):
    
    async def event_stream():
        while True:
            if await raw_request.is_disconnected():
                print("Client disconnected, stopping generation loop.")
                break

            # (Provider selection logic remains the same)
            if req.provider == 'ollama':
                generator = llm_generator.generate_with_ollama(
                    model=req.model, 
                    label_to_generate=req.label_to_generate
                )
            # ... (other providers)
            else:
                yield "data: Error: Invalid provider specified.\n\n"
                break
            
            try:
                async for status_or_data in generator:
                    # The generator now yields either a status string or a data dictionary.
                    if isinstance(status_or_data, dict):
                        # It's our final data object!
                        data = status_or_data
                        print(f"✅ [LLM Generated] Label: {data['label']:<4} | Message: {data['message']}")
                        database.add_feedback(data['message'], data['label'], source='llm')
                        # Format the data for display on the dashboard
                        yield f"data: Generated & Saved: {json.dumps(data)}\n\n"
                    else:
                        # It's a regular status string (e.g., "Thinking...")
                        yield f"data: {status_or_data}\n\n"
                    
                    await asyncio.sleep(0.1)

                yield "data: Pausing for 1.5 seconds...\n\n"
                await asyncio.sleep(0.2)  # Pause to simulate processing time

            except Exception as e:
                error_message = f"An error occurred in the generation loop: {e}"
                print(error_message)
                yield f"data: {error_message}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")