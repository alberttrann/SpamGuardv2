# backend/main.py (Final, Definitive, Fully Corrected Version)

from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict
import threading
import os
import time
import json
import numpy as np
from fastapi.responses import StreamingResponse
import asyncio

# Use relative imports for local modules
from .classifier import SpamGuardClassifier
from . import database
from . import llm_generator
from .train_nb import retrain_and_save
from . import registry

app = FastAPI(title="SpamGuard AI API", version="4.1.0") # Final version bump

# --- CONSOLIDATED STATE MANAGEMENT ---
class AppStateManager:
    """A single, thread-safe class to manage all application state."""
    def __init__(self):
        self.prod_classifier = SpamGuardClassifier()
        self.is_loading_model = False
        self.model_status_message = "Idle"
        self.is_generating_llm = False
        self.llm_generated_data = []
        self.llm_status_message = "Idle"
        self._lock = threading.Lock()

manager = AppStateManager()

# --- Background Task Functions ---
def load_prod_classifier():
    """Target function for background thread to load/reload the main classifier."""
    with manager._lock:
        if manager.is_loading_model: return
        manager.is_loading_model = True
        manager.model_status_message = "Loading model configuration in background..."
    manager.prod_classifier.load()
    with manager._lock:
        manager.is_loading_model = False
        manager.model_status_message = "System is ready."
    print("BACKGROUND: Production model is now loaded and ready.")

def run_retraining_sequence():
    """This function handles the entire long-running retraining process."""
    with manager._lock:
        if manager.is_loading_model: return
        manager.is_loading_model = True
        manager.model_status_message = "Step 1/3: Enriching dataset..."
    
    config = registry.get_current_config()
    dataset_file = config.get("knn_dataset_file")
    if not dataset_file:
        with manager._lock: manager.model_status_message = "Error: No dataset configured."; manager.is_loading_model = False; return
        
    new_records_count = database.enrich_main_dataset(dataset_file)
    if new_records_count == 0:
        with manager._lock: manager.model_status_message = "Retraining skipped: No new feedback."; manager.is_loading_model = False; return

    with manager._lock: manager.model_status_message = f"Step 2/3: Training new model..."
    retrain_and_save(dataset_file)

    with manager._lock: manager.model_status_message = "Step 3/3: Loading new model..."
    manager.prod_classifier.reload()

    with manager._lock:
        manager.is_loading_model = False
        manager.model_status_message = "Retraining complete. New model is active."
    print("BACKGROUND: Retraining sequence finished successfully.")

# --- THIS IS THE CORRECTED, CONTINUOUS LLM GENERATION LOOP ---
async def run_llm_generation_in_background(provider, model, api_key, label_to_generate):
    """The async target for our continuous background generation task."""
    with manager._lock:
        if manager.is_generating_llm: return
        manager.is_generating_llm = True
        manager.llm_status_message = "Starting generation..."
        manager.llm_generated_data.clear()
    
    generator_map = {'ollama': llm_generator.generate_with_ollama, 'lmstudio': llm_generator.generate_with_lmstudio, 'openrouter': llm_generator.generate_with_openrouter}
    generator_func = generator_map.get(provider)
    if not generator_func:
        with manager._lock: manager.llm_status_message = "Error: Invalid provider."; manager.is_generating_llm = False; return

    kwargs = {"model": model, "label_to_generate": label_to_generate}
    if provider == 'openrouter':
        if not api_key:
            with manager._lock: manager.llm_status_message = "Error: API key required."; manager.is_generating_llm = False; return
        kwargs["api_key"] = api_key
    
    # This while loop ensures continuous generation until stopped.
    while manager.is_generating_llm:
        async for item in generator_func(**kwargs):
            # The inner loop runs once per generation, which is intended.
            with manager._lock:
                if not manager.is_generating_llm: break # Check flag again
                if isinstance(item, dict):
                    # DO NOT save to DB. Add to temporary review list.
                    manager.llm_generated_data.append(item)
                    manager.llm_status_message = f"Generated {len(manager.llm_generated_data)} messages..."
                else:
                    manager.llm_status_message = item # This is a status string
        
        if not manager.is_generating_llm: break # Check flag before sleeping
        await asyncio.sleep(0.5) # Small delay between generating each message
        
    with manager._lock:
        manager.llm_status_message = f"Generation stopped. {len(manager.llm_generated_data)} messages ready for review."
        manager.is_generating_llm = False

# --- FastAPI Events ---
@app.on_event("startup")
def startup_event():
    database.init_db()
    reg = registry.get_all_models()
    if not reg.get("active_model_id") and reg.get("models"):
        latest_id = sorted(reg["models"].items(), key=lambda i: i[1]['creation_date'], reverse=True)[0][0]
        registry.set_active_model(latest_id)
    print("Spawning background thread for initial model load.")
    threading.Thread(target=load_prod_classifier, daemon=True).start()

# --- Pydantic Models ---
class Message(BaseModel): text: str
class Feedback(BaseModel): message: str; correct_label: str
class LLMRequest(BaseModel): provider: str; model: str; api_key: str | None = None; label_to_generate: str | None = None
class BulkFeedbackItem(BaseModel): label: str; message: str
class BulkMessageRequest(BaseModel): messages: List[str]    
class ActivateModelRequest(BaseModel): model_id: str
class ConfigRequest(BaseModel): mode: str; knn_dataset_file: str
class DeleteFeedbackRequest(BaseModel): ids: List[int]

# --- API Endpoints ---
@app.get("/status")
def get_status():
    with manager._lock: return { "is_ready": manager.prod_classifier.is_loaded, "is_loading_new_config": manager.is_loading_model, "loading_status_message": manager.model_status_message }

@app.get("/")
def read_root(): return {"message": "Welcome to the SpamGuard AI API."}

@app.post("/classify")
def classify_message(message: Message):
    if not manager.prod_classifier.is_loaded: return {"error": "Classifier is not ready yet."}
    return manager.prod_classifier.classify(message.text)

@app.post("/feedback")
def receive_feedback(feedback: Feedback):
    database.add_feedback(message=feedback.message, label=feedback.correct_label, source='user')
    return {"status": "success", "message": "Feedback received."}

@app.post("/bulk_feedback")
def receive_bulk_feedback(feedback_list: List[BulkFeedbackItem]):
    for item in feedback_list: database.add_feedback(message=item.message, label=item.label, source='user')
    return {"status": "success", "message": f"Successfully added {len(feedback_list)} records."}

@app.get("/analytics")
def get_analytics():
    current_config = registry.get_current_config()
    dataset_file = current_config.get("knn_dataset_file", "2cls_spam_text_cls.csv")
    return database.get_analytics(dataset_file)

@app.post("/retrain")
def trigger_retraining(background_tasks: BackgroundTasks):
    if manager.is_loading_model: return {"status": "error", "message": "Cannot start retraining while another task is in progress."}
    background_tasks.add_task(run_retraining_sequence)
    return {"status": "success", "message": "Retraining process started in the background."}

# --- Model and Config Management Endpoints ---
@app.get("/models")
def list_models(): return registry.get_all_models()

@app.post("/models/activate")
def activate_model(req: ActivateModelRequest, background_tasks: BackgroundTasks):
    if manager.is_loading_model: return {"status": "error", "message": "Cannot activate a new model while another task is in progress."}
    registry.set_active_model(req.model_id)
    background_tasks.add_task(load_prod_classifier)
    return {"status": "success", "message": f"Model '{req.model_id}' set to active. Reloading in background."}

@app.get("/config")
def get_config(): return registry.get_current_config()

@app.post("/config")
def set_config(req: ConfigRequest, background_tasks: BackgroundTasks):
    if manager.is_loading_model: return {"status": "error", "message": "Cannot change configuration while another task is in progress."}
    registry.set_current_config(req.mode, req.knn_dataset_file)
    background_tasks.add_task(load_prod_classifier)
    return {"status": "success", "message": "New configuration update started in the background."}

# --- Classifier Intelligence Endpoints ---
@app.get("/explain_model")
def explain_nb_model(top_n: int = 20):
    if not manager.prod_classifier.is_loaded: return {"error": "Classifier is not ready yet."}
    return manager.prod_classifier.explain_model(top_n=top_n)

@app.post("/get_nb_probabilities")
def get_nb_probabilities(req: BulkMessageRequest):
    if not manager.prod_classifier.is_loaded: return {"error": "Classifier is not ready yet."}
    return manager.prod_classifier.get_nb_probabilities(req.messages)

# --- LLM Endpoints ---
@app.post("/llm/start_generation")
def start_llm_generation(req: LLMRequest, background_tasks: BackgroundTasks):
    if manager.is_generating_llm: return {"status": "error", "message": "A generation task is already in progress."}
    background_tasks.add_task(lambda: asyncio.run(run_llm_generation_in_background(req.provider, req.model, req.api_key, req.label_to_generate)))
    return {"status": "success", "message": "LLM data generation started in the background."}

@app.post("/llm/stop_generation")
def stop_llm_generation():
    if not manager.is_generating_llm: return {"status": "skipped", "message": "No generation task is currently running."}
    with manager._lock: manager.is_generating_llm = False
    return {"status": "success", "message": "Stop signal sent to the generation task."}

@app.get("/llm/review_data")
def get_llm_review_data():
    with manager._lock:
        return { "is_generating": manager.is_generating_llm, "status_message": manager.llm_status_message, "data": manager.llm_generated_data }

@app.post("/llm/clear_review_data")
def clear_llm_review_data():
    with manager._lock: manager.llm_generated_data.clear()
    return {"status": "success", "message": "Review data cleared."}

# --- Feedback Staging Area Endpoints ---
@app.get("/feedback/all")
def get_all_pending_feedback(): return database.get_all_feedback()

@app.post("/feedback/delete")
def delete_pending_feedback(req: DeleteFeedbackRequest):
    deleted_count = database.delete_feedback_by_ids(req.ids)
    return {"status": "success", "message": f"Deleted {deleted_count} records."}

# --- THIS IS THE NEW ENDPOINT ---
@app.post("/feedback/delete_all")
def delete_all_pending_feedback():
    deleted_count = database.delete_all_feedback()
    return {"status": "success", "message": f"Discarded all {deleted_count} records from the staging area."}