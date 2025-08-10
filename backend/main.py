# backend/main.py (Final Version with Hot-Swapping and Background Tasks)

from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict
import threading
import os
import time
import json
import numpy as np
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
import asyncio

# Use relative imports for local modules
from .classifier import SpamGuardClassifier
from . import database
from . import llm_generator
from .train_nb import retrain_and_save
from . import registry

app = FastAPI(title="SpamGuard AI API", version="3.2.0")

# --- NEW: Dual Classifier Singleton Pattern for Hot-Swapping ---
class ClassifierManager:
    """A thread-safe manager to handle production and staging classifier instances."""
    def __init__(self):
        self.prod_classifier = SpamGuardClassifier() # The one currently serving traffic
        self.staging_classifier = None # The one being loaded in the background
        self.is_loading = False
        self.status_message = "Idle"
        self._lock = threading.Lock() # To prevent race conditions during the swap

manager = ClassifierManager()

def load_and_swap_classifier():
    """
    The target function for our background thread. This function performs the
    entire slow loading process and then atomically swaps the new classifier
    into production.
    """
    with manager._lock:
        if manager.is_loading:
            print("BACKGROUND: A load process is already running. Exiting new request.")
            return
        manager.is_loading = True
        manager.status_message = "Loading new model configuration in background..."
    
    # Create and load a new instance in staging
    staging_instance = SpamGuardClassifier()
    staging_instance.load() # This is the long, blocking call
    
    with manager._lock:
        # --- The Atomic Swap ---
        # Once loading is complete, replace the production instance
        manager.prod_classifier = staging_instance
        manager.is_loading = False
        manager.status_message = "Configuration applied successfully. System is ready."
    print("BACKGROUND: New model configuration loaded and swapped to production.")

@app.on_event("startup")
def startup_event():
    """

    On startup, immediately spawns a background thread to load the initial models.
    The main server thread is NOT blocked and can serve requests instantly.
    """
    database.init_db()
    reg = registry.get_all_models()
    if not reg.get("active_model_id") and reg.get("models"):
        latest_id = sorted(reg["models"].items(), key=lambda i: i[1]['creation_date'], reverse=True)[0][0]
        registry.set_active_model(latest_id)
        print(f"No active model found. Automatically set '{latest_id}' as active.")
    
    # Start the initial load in a separate thread
    print("Spawning background thread for initial model load.")
    threading.Thread(target=load_and_swap_classifier, daemon=True).start()


# --- Pydantic Models ---
class Message(BaseModel): text: str
class Feedback(BaseModel): message: str; correct_label: str
class LLMRequest(BaseModel): provider: str; model: str; api_key: str | None = None; label_to_generate: str | None = None
class BulkFeedbackItem(BaseModel): label: str; message: str
class BulkMessageRequest(BaseModel): messages: List[str]    
class ActivateModelRequest(BaseModel): model_id: str
class ConfigRequest(BaseModel): mode: str; knn_dataset_file: str

# --- API Endpoints ---

@app.get("/status")
def get_status():
    """Returns the readiness of the production classifier and any loading status."""
    with manager._lock:
        return {
            "is_ready": manager.prod_classifier.is_loaded,
            "is_loading_new_config": manager.is_loading,
            "loading_status_message": manager.status_message
        }

@app.get("/")
def read_root(): return {"message": "Welcome to the SpamGuard AI API."}

@app.post("/classify")
def classify_message(message: Message):
    if not manager.prod_classifier.is_loaded:
        return {"error": "Classifier is not ready yet. Please wait for the initial load to complete."}
    return manager.prod_classifier.classify(message.text)

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
    
    # After retraining, we MUST tell the user to restart the loader process.
    # The current instance is now stale.
    with manager._lock:
        manager.prod_classifier.is_loaded = False
        manager.status_message = "Model retrained. Restart the server and loader to activate the new model."
    
    return {"status": "success", "message": f"Retraining complete with {new_records_count} new records. Please restart the backend server to apply the new model."}

@app.get("/models")
def list_models():
    return registry.get_all_models()

@app.post("/models/activate")
def activate_model(req: ActivateModelRequest, background_tasks: BackgroundTasks):
    if manager.is_loading:
        return {"status": "error", "message": "Cannot activate a new model while another update is in progress."}
    registry.set_active_model(req.model_id)
    background_tasks.add_task(load_and_swap_classifier)
    return {"status": "success", "message": f"Model '{req.model_id}' set to active. Reloading in background."}

@app.get("/config")
def get_config():
    return registry.get_current_config()

@app.post("/config")
def set_config(req: ConfigRequest, background_tasks: BackgroundTasks):
    if manager.is_loading:
        return {"status": "error", "message": "Cannot change configuration while another update is in progress."}
    registry.set_current_config(req.mode, req.knn_dataset_file)
    background_tasks.add_task(load_and_swap_classifier)
    return {"status": "success", "message": "New configuration update started in the background."}

@app.get("/explain_model")
def explain_nb_model(top_n: int = 20):
    if not manager.prod_classifier.is_loaded:
        return {"error": "Classifier is not ready yet."}
    return manager.prod_classifier.explain_model(top_n=top_n)

@app.post("/get_nb_probabilities")
def get_nb_probabilities(req: BulkMessageRequest):
    if not manager.prod_classifier.is_loaded:
        return {"error": "Classifier is not ready yet."}
    return manager.prod_classifier.get_nb_probabilities(req.messages)

@app.post("/generate_data")
async def generate_data_stream(req: LLMRequest, raw_request: Request):
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