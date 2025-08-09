# backend/main.py (Final, Definitive Non-Blocking Version)

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from fastapi.concurrency import run_in_threadpool
import json
import os
from fastapi.responses import StreamingResponse
import asyncio

# Use relative imports for local modules
from .classifier import SpamGuardClassifier
from . import database
from . import llm_generator
from .train_nb import retrain_and_save
from . import registry

app = FastAPI(title="SpamGuard AI API", version="3.0.0") # Final version bump

# The singleton is still created lazy, but it will be loaded by the loader script
# or by the first /status call after the flag file is present.
classifier_singleton = SpamGuardClassifier() 

# Define the path for our communication flag file
FLAG_FILE_PATH = os.path.join(os.path.dirname(__file__), "_ready.flag")

@app.on_event("startup")
def startup_event():
    """
    Startup is now minimal. It only initializes the database.
    The heavy model loading is handled by a separate process.
    """
    database.init_db()
    print("Application startup complete. Waiting for loader script to signal readiness via flag file.")

# --- Pydantic Models ---
class Message(BaseModel): text: str
class Feedback(BaseModel): message: str; correct_label: str
class LLMRequest(BaseModel): provider: str; model: str; api_key: str | None = None; label_to_generate: str | None = None
class BulkFeedbackItem(BaseModel): label: str; message: str
class BulkMessageRequest(BaseModel): messages: List[str]    
class ActivateModelRequest(BaseModel): model_id: str

# --- API Endpoints ---

@app.get("/status")
def get_status():
    """
    A fast, lightweight endpoint that checks if the classifier is ready.
    It checks for the existence of the flag file created by loader.py.
    """
    is_ready = os.path.exists(FLAG_FILE_PATH)
    
    # If the flag exists but the in-memory object isn't loaded yet (e.g., after a --reload), load it.
    # This is a one-time sync step that happens only if the flag is already present.
    if is_ready and not classifier_singleton.is_loaded:
        print("Ready flag detected, but classifier not loaded in this instance. Loading now...")
        classifier_singleton.load()
        
    return {"is_ready": classifier_singleton.is_loaded}

@app.get("/")
def read_root(): return {"message": "Welcome to the SpamGuard AI API. Use the /status endpoint to check readiness."}

@app.post("/classify")
def classify_message(message: Message):
    if not classifier_singleton.is_loaded:
        return {"error": "Classifier is not ready. Please ensure the loader script has completed."}
    # This is now a fast, synchronous call because the model is already loaded.
    return classifier_singleton.classify(message.text)

@app.post("/feedback")
def receive_feedback(feedback: Feedback):
    database.add_feedback(message=feedback.message, label=feedback.correct_label)
    return {"status": "success", "message": "Feedback received."}

@app.post("/bulk_feedback")
def receive_bulk_feedback(feedback_list: List[BulkFeedbackItem]):
    for item in feedback_list: database.add_feedback(message=item.message, label=item.label, source='user')
    return {"status": "success", "message": f"Successfully added {len(feedback_list)} records."}

@app.get("/analytics")
def get_analytics():
    return database.get_analytics()

@app.post("/retrain")
async def trigger_retraining():
    # Retraining is a long process, so we run it in a threadpool to not block the server.
    print("Retraining process started in background...")
    new_records_count = await run_in_threadpool(database.enrich_main_dataset)
    if new_records_count == 0:
        return {"status": "skipped", "message": "No new feedback data to train on."}
    
    await run_in_threadpool(retrain_and_save)
    
    # After retraining, we MUST delete the flag file to signal the model is stale.
    if os.path.exists(FLAG_FILE_PATH):
        os.remove(FLAG_FILE_PATH)
        
    # Mark the current instance as not loaded. A new run of loader.py is required.
    classifier_singleton.is_loaded = False
    print("Retraining complete. The ready flag has been removed. Please run the loader script again.")
    
    return {"status": "success", "message": f"Retraining complete with {new_records_count} new records. Please run the loader script to activate the new model."}

@app.get("/models")
def list_models():
    return registry.get_all_models()

@app.post("/models/activate")
def activate_model(req: ActivateModelRequest):
    # Activating a model is just changing the registry. The loader will pick it up.
    try:
        registry.set_active_model(req.model_id)
        # The model is now stale. Remove the flag and require a new load.
        if os.path.exists(FLAG_FILE_PATH):
            os.remove(FLAG_FILE_PATH)
        classifier_singleton.is_loaded = False
        return {"status": "success", "message": f"Model '{req.model_id}' set as active. Please run the loader script to load it."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/explain_model")
def explain_nb_model(top_n: int = 20):
    if not classifier_singleton.is_loaded:
        return {"error": "Classifier is not ready."}
    return classifier_singleton.explain_model(top_n=top_n)

@app.post("/get_nb_probabilities")
def get_nb_probabilities(req: BulkMessageRequest):
    if not classifier_singleton.is_loaded:
        return {"error": "Classifier is not ready."}
    return classifier_singleton.get_nb_probabilities(req.messages)

@app.post("/generate_data")
async def generate_data_stream(req: LLMRequest, raw_request: Request):
    # This endpoint is already async and streaming, it's fine as is.
    async def event_stream():
        while True:
            if await raw_request.is_disconnected(): break
            generator = None
            if req.provider == 'ollama': generator = llm_generator.generate_with_ollama(model=req.model, label_to_generate=req.label_to_generate)
            elif req.provider == 'lmstudio': generator = llm_generator.generate_with_lmstudio(model=req.model, label_to_generate=req.label_to_generate)
            elif req.provider == 'openrouter':
                if not req.api_key: yield "data: Error: API key required.\n\n"; break
                generator = llm_generator.generate_with_openrouter(model=req.model, api_key=req.api_key, label_to_generate=req.label_to_generate)
            else: yield "data: Error: Invalid provider specified.\n\n"; break
            try:
                async for item in generator:
                    if isinstance(item, dict):
                        database.add_feedback(item['message'], item['label'], source='llm')
                        yield f"data: Generated & Saved: {json.dumps(item)}\n\n"
                    else: yield f"data: {item}\n\n"
                    await asyncio.sleep(0.1)
                yield "data: Pausing for 1.5 seconds...\n\n"; await asyncio.sleep(1.5)
            except Exception as e: yield f"data: An error occurred: {e}\n\n"; break
    return StreamingResponse(event_stream(), media_type="text/event-stream")