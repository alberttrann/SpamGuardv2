# backend/registry.py (Final, Definitive, Fully-Featured Version)

import os
import json
from datetime import datetime

# --- Paths ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(BACKEND_DIR, 'data')
MODEL_REGISTRY_PATH = os.path.join(MODELS_DIR, 'registry.json')
DATA_REGISTRY_PATH = os.path.join(DATA_DIR, 'data_registry.json') # New registry for datasets


# --- Model Registry Functions ---

def _get_model_registry():
    """Reads the model registry, initializing with defaults if not found or corrupt."""
    if not os.path.exists(MODEL_REGISTRY_PATH):
        return {
            "models": {},
            "active_model_id": None,
            "current_config": {
                "mode": "hybrid",
                "knn_dataset_file": "2cls_spam_text_cls.csv"
            }
        }
    try:
        with open(MODEL_REGISTRY_PATH, 'r') as f:
            data = json.load(f)
            # Backwards compatibility: ensure new fields exist for old registry files
            if "current_config" not in data:
                data["current_config"] = {
                    "mode": "hybrid",
                    "knn_dataset_file": "2cls_spam_text_cls.csv"
                }
            for model_id in data["models"]:
                if "note" not in data["models"][model_id]:
                    data["models"][model_id]["note"] = ""
            return data
    except (json.JSONDecodeError, IOError):
        print("Warning: Model registry file corrupted. Starting fresh.")
        return {
            "models": {}, "active_model_id": None,
            "current_config": {"mode": "hybrid", "knn_dataset_file": "2cls_spam_text_cls.csv"}
        }

def _save_model_registry(data):
    """Saves the model registry data to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_REGISTRY_PATH, 'w') as f:
        json.dump(data, f, indent=4)

def add_model_to_registry(model_id: str, pipeline_filename: str, encoder_filename: str):
    """Adds a new model's metadata to the registry, including the 'note' field."""
    reg = _get_model_registry()
    reg["models"][model_id] = {
        "creation_date": datetime.now().isoformat(),
        "pipeline_file": pipeline_filename,
        "encoder_file": encoder_filename,
        "note": ""  # Initialize with an empty note
    }
    _save_model_registry(reg)
    print(f"Model '{model_id}' added to registry.")

def update_model_notes(notes: dict):
    """Updates the 'note' field for multiple models in the registry."""
    reg = _get_model_registry()
    for model_id, note in notes.items():
        if model_id in reg["models"]:
            reg["models"][model_id]["note"] = note
    _save_model_registry(reg)

def get_all_models():
    """Returns all model metadata from the registry."""
    return _get_model_registry()

def set_active_model(model_id: str):
    """Sets a given model ID as the active model for the application."""
    reg = _get_model_registry()
    if model_id not in reg["models"]:
        raise ValueError(f"Model ID '{model_id}' not found in registry.")
    reg["active_model_id"] = model_id
    _save_model_registry(reg)
    print(f"Model '{model_id}' is now the active model.")

def get_active_model_paths():
    """Returns the file paths for the currently active model."""
    reg = _get_model_registry()
    active_id = reg.get("active_model_id")
    if not active_id or active_id not in reg["models"]:
        return None, None
    model_info = reg["models"][active_id]
    pipeline_path = os.path.join(MODELS_DIR, model_info["pipeline_file"])
    encoder_path = os.path.join(MODELS_DIR, model_info["encoder_file"])
    return pipeline_path, encoder_path


# --- NEW: Data Registry Functions ---

def _get_data_registry():
    """Reads the data registry file. Returns an empty dict if not found."""
    if not os.path.exists(DATA_REGISTRY_PATH):
        return {}  # e.g., { "filename.csv": {"note": ""} }
    try:
        with open(DATA_REGISTRY_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def _save_data_registry(data):
    """Saves the data registry to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_REGISTRY_PATH, 'w') as f:
        json.dump(data, f, indent=4)

def get_data_registry_with_files():
    """Scans the data directory for CSV files and syncs them with the registry."""
    reg = _get_data_registry()
    try:
        found_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    except FileNotFoundError:
        found_files = []
        
    # Add any new CSV files found on disk to the registry
    for f_name in found_files:
        if f_name not in reg:
            reg[f_name] = {"note": ""}
            
    # Remove any files from the registry that no longer exist on disk
    for f_name in list(reg.keys()):
        if f_name not in found_files:
            del reg[f_name]
            
    _save_data_registry(reg)
    return reg

def update_dataset_notes(notes: dict):
    """Updates the 'note' field for multiple datasets in the registry."""
    reg = _get_data_registry()
    for filename, note in notes.items():
        if filename in reg:
            reg[filename]["note"] = note
    _save_data_registry(reg)


# --- Config functions that use the Model Registry ---

def get_current_config():
    """Returns the current classifier configuration from the main registry."""
    reg = _get_model_registry()
    return reg.get("current_config")

def set_current_config(mode: str, knn_dataset_file: str):
    """Sets the classifier's operational mode and k-NN dataset in the main registry."""
    reg = _get_model_registry()
    if "current_config" not in reg:
        reg["current_config"] = {}
    reg["current_config"]["mode"] = mode
    reg["current_config"]["knn_dataset_file"] = knn_dataset_file
    _save_model_registry(reg)
    print(f"Classifier config updated: Mode='{mode}', k-NN Dataset='{knn_dataset_file}'")