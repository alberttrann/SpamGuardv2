# backend/main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import asyncio
import json
import numpy as np
from typing import List

# Use relative imports for local modules
from .classifier import SpamGuardClassifier
from . import database
from . import llm_generator
from .train_nb import retrain_and_save
from . import registry

app = FastAPI(title="SpamGuard AI API", version="2.3.0")

@app.on_event("startup")
def startup_event():
    """
    Initializes the database and ensures there's an active model.
    If no model is active, it tries to set the most recent one.
    """
    database.init_db()
    
    reg = registry.get_all_models()
    active_id = reg.get("active_model_id")
    all_models = reg.get("models")

    if not active_id and all_models:
        print("No active model set. Activating the most recent model...")
        latest_model_id = sorted(all_models.items(), key=lambda item: item[1]['creation_date'], reverse=True)[0][0]
        registry.set_active_model(latest_model_id)
        print(f"Model '{latest_model_id}' automatically activated.")

    app.state.classifier = SpamGuardClassifier()


# --- Pydantic Models ---
class Message(BaseModel): text: str
class Feedback(BaseModel): message: str; correct_label: str
class LLMRequest(BaseModel): provider: str; model: str; api_key: str | None = None; label_to_generate: str | None = None
class BulkFeedbackItem(BaseModel): label: str; message: str
class BulkMessageRequest(BaseModel): messages: List[str]    
class ActivateModelRequest(BaseModel): model_id: str


# --- API Endpoints ---
@app.get("/")
def read_root(): return {"message": "Welcome to the SpamGuard AI API"}

@app.post("/classify")
def classify_message(message: Message):
    return app.state.classifier.classify(message.text)

@app.post("/feedback")
def receive_feedback(feedback: Feedback):
    database.add_feedback(message=feedback.message, label=feedback.correct_label)
    return {"status": "success", "message": "Feedback received."}

@app.post("/bulk_feedback")
def receive_bulk_feedback(feedback_list: List[BulkFeedbackItem]):
    try:
        for item in feedback_list:
            database.add_feedback(message=item.message, label=item.label, source='user')
        return {"status": "success", "message": f"Successfully added {len(feedback_list)} records."}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {e}"}

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

@app.get("/models")
def list_models():
    return registry.get_all_models()

@app.post("/models/activate")
def activate_model(req: ActivateModelRequest):
    try:
        registry.set_active_model(req.model_id)
        app.state.classifier.reload()
        return {"status": "success", "message": f"Model '{req.model_id}' activated successfully."}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}

@app.get("/explain_model")
def explain_nb_model(top_n: int = 20):
    classifier_instance = app.state.classifier
    if not classifier_instance.nb_pipeline:
        return {"error": "Naive Bayes model not loaded."}
    try:
        vectorizer = classifier_instance.nb_pipeline.named_steps['tfidf']
        model = classifier_instance.nb_pipeline.named_steps['clf']
        label_encoder = classifier_instance.label_encoder
        feature_names = np.array(vectorizer.get_feature_names_out())
        log_probs = model.feature_log_prob_
        spam_idx = np.where(label_encoder.classes_ == 'spam')[0][0]
        ham_idx = np.where(label_encoder.classes_ == 'ham')[0][0]
        top_spam_indices = log_probs[spam_idx].argsort()[-top_n:][::-1]
        top_ham_indices = log_probs[ham_idx].argsort()[-top_n:][::-1]
        return {
            "top_spam_keywords": feature_names[top_spam_indices].tolist(),
            "top_ham_keywords": feature_names[top_ham_indices].tolist()
        }
    except Exception as e:
        return {"error": f"An error occurred during explanation: {e}"}

@app.post("/get_nb_probabilities")
def get_nb_probabilities(req: BulkMessageRequest):
    classifier_instance = app.state.classifier
    if not classifier_instance.nb_pipeline:
        return {"error": "Naive Bayes model not loaded."}
    spam_idx = np.where(classifier_instance.label_encoder.classes_ == 'spam')[0][0]
    all_probs = classifier_instance.nb_pipeline.predict_proba(req.messages)
    return {"spam_probabilities": [p[spam_idx] for p in all_probs]}

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