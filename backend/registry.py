# backend/registry.py

import os
import json
from datetime import datetime

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REGISTRY_PATH = os.path.join(MODELS_DIR, 'registry.json')

def _get_registry():
    """Reads the registry file from disk. Initializes default config if not found."""
    if not os.path.exists(REGISTRY_PATH):
        return {
            "models": {},
            "active_model_id": None,
            "current_config": { # --- NEW: Default configuration ---
                "mode": "hybrid", # 'hybrid', 'nb_only', 'knn_only'
                "knn_dataset_file": "2cls_spam_text_cls.csv" # Default to latest data
            }
        }
    try:
        with open(REGISTRY_PATH, 'r') as f:
            data = json.load(f)
            # Ensure new config fields exist for old registry files
            if "current_config" not in data:
                data["current_config"] = {
                    "mode": "hybrid",
                    "knn_dataset_file": "2cls_spam_text_cls.csv"
                }
            return data
    except (json.JSONDecodeError, IOError):
        print("Warning: Registry file corrupted or unreadable. Starting fresh.")
        return {
            "models": {},
            "active_model_id": None,
            "current_config": {
                "mode": "hybrid",
                "knn_dataset_file": "2cls_spam_text_cls.csv"
            }
        }

def _save_registry(registry_data):
    """Saves the registry data to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry_data, f, indent=4)

def add_model_to_registry(model_id: str, pipeline_filename: str, encoder_filename: str):
    """Adds a new model's metadata to the registry."""
    registry = _get_registry()
    
    new_model_entry = {
        "creation_date": datetime.now().isoformat(),
        "pipeline_file": pipeline_filename,
        "encoder_file": encoder_filename,
        "performance_metrics": None 
    }
    
    registry["models"][model_id] = new_model_entry
    _save_registry(registry)
    print(f"Model '{model_id}' added to registry.")

def set_active_model(model_id: str):
    """Sets a given model ID as the active model for the application."""
    registry = _get_registry()
    if model_id not in registry["models"]:
        raise ValueError(f"Model ID '{model_id}' not found in registry.")
    
    registry["active_model_id"] = model_id
    _save_registry(registry)
    print(f"Model '{model_id}' is now the active model.")

def get_active_model_paths():
    """
    Returns the file paths for the currently active model.
    Returns (None, None) if no active model is set.
    """
    registry = _get_registry()
    active_id = registry.get("active_model_id")
    
    if not active_id or active_id not in registry["models"]:
        return None, None
        
    model_info = registry["models"][active_id]
    pipeline_path = os.path.join(MODELS_DIR, model_info["pipeline_file"])
    encoder_path = os.path.join(MODELS_DIR, model_info["encoder_file"])
    
    return pipeline_path, encoder_path

def get_all_models():
    """Returns all model metadata from the registry."""
    return _get_registry()

def get_current_config():
    """Returns the current classifier configuration."""
    registry = _get_registry()
    return registry["current_config"]

def set_current_config(mode: str, knn_dataset_file: str):
    """Sets the classifier's operational mode and k-NN dataset."""
    registry = _get_registry()
    if mode not in ["hybrid", "nb_only", "knn_only"]:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'hybrid', 'nb_only', or 'knn_only'.")
    
    data_dir_path = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(os.path.join(data_dir_path, knn_dataset_file)):
        print(f"Warning: Selected k-NN dataset file '{knn_dataset_file}' not found in '{data_dir_path}'.")

    registry["current_config"]["mode"] = mode
    registry["current_config"]["knn_dataset_file"] = knn_dataset_file
    _save_registry(registry)
    print(f"Classifier config updated: Mode='{mode}', k-NN Dataset='{knn_dataset_file}'")