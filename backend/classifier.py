# backend/classifier.py (Final, Definitive Version)

import joblib
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import os
from typing import List, Dict

# Use relative imports for local modules
from .utils import preprocess_tokenizer
from . import registry

# --- Define robust absolute paths ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(BACKEND_DIR, 'data')


class SpamGuardClassifier:
    """
    A dynamic, hybrid classifier that can operate in multiple modes.
    It uses lazy loading and intelligent caching for high performance.
    """
    def __init__(self):
        """
        Initializes the classifier in a 'lazy' state. Models are not loaded.
        """
        print("SpamGuardClassifier instance created in a lazy state.")
        self.is_loaded = False
        self.nb_pipeline = None
        self.label_encoder = None
        self.transformer_model = None
        self.tokenizer = None
        self.device = None
        self.faiss_index = None
        self.all_messages = []
        self.all_labels = []

    def load(self):
        """
        Loads classifier components based on the active configuration from the registry.
        """
        print("--- LAZY LOADING: Initializing SpamGuard AI Classifier with current config ---")
        self.is_loaded = False # Reset loaded state for a fresh load

        current_config = registry.get_current_config()
        mode = current_config.get("mode", "hybrid")
        knn_dataset_file = current_config.get("knn_dataset_file")

        print(f"Loading in '{mode}' mode. k-NN dataset: '{knn_dataset_file}'")

        # --- 1. Load Naive Bayes Pipeline (Conditional) ---
        if mode in ["hybrid", "nb_only"]:
            pipeline_path, encoder_path = registry.get_active_model_paths()
            if pipeline_path and encoder_path and os.path.exists(pipeline_path):
                try:
                    self.nb_pipeline = joblib.load(pipeline_path)
                    self.label_encoder = joblib.load(encoder_path)
                    active_id = registry.get_all_models().get("active_model_id", "N/A")
                    print(f"✅ Active model '{active_id}' loaded for Naive Bayes.")
                except Exception as e:
                    print(f"🔴 ERROR loading active NB model: {e}. NB will be skipped.")
                    self.nb_pipeline = None; self.label_encoder = None
            else:
                print(f"🔴 WARNING: No active Naive Bayes model found in registry for current mode.")
                self.nb_pipeline = None; self.label_encoder = None
        else:
            self.nb_pipeline = None; self.label_encoder = None
            print("INFO: Skipping Naive Bayes load (mode is k-NN only).")

        # --- 2. Load Transformer Model (Conditional) ---
        if mode in ["hybrid", "knn_only"]:
            if self.transformer_model is None: 
                MODEL_NAME = "intfloat/multilingual-e5-base"
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.transformer_model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()
                print(f"✅ Transformer model loaded on {self.device}.")
        else:
            self.transformer_model = None
            print("INFO: Skipping Transformer load (mode is NB only).")


        # --- 3. Build/Load FAISS Index (Conditional) ---
        if mode in ["hybrid", "knn_only"]:
            knn_data_path = os.path.join(DATA_DIR, knn_dataset_file)
            
            if not os.path.exists(knn_data_path):
                print(f"🔴 WARNING: k-NN dataset file '{knn_dataset_file}' not found. FAISS index cannot be built.")
                self.all_messages = []; self.all_labels = []; self.faiss_index = None
            else:
                df_knn_data = pd.read_csv(knn_data_path, quotechar='"', on_bad_lines='skip')
                df_knn_data.dropna(subset=['Message', 'Category'], inplace=True)
                self.all_messages = df_knn_data["Message"].astype(str).tolist()
                self.all_labels = df_knn_data["Category"].tolist()
                
                faiss_index_filename = f"faiss_index_{knn_dataset_file.replace('.', '_')}.bin"
                FAISS_INDEX_CACHED_PATH = os.path.join(MODELS_DIR, faiss_index_filename)
                
                cache_is_valid = (
                    os.path.exists(FAISS_INDEX_CACHED_PATH) and
                    os.path.getmtime(FAISS_INDEX_CACHED_PATH) >= os.path.getmtime(knn_data_path)
                )
                
                if cache_is_valid:
                    print(f"✅ Found valid cache for '{knn_dataset_file}'. Loading FAISS index from disk...")
                    self.faiss_index = faiss.read_index(FAISS_INDEX_CACHED_PATH)
                    print("✅ FAISS index loaded from cache.")
                else:
                    print(f"🔴 Cache for '{knn_dataset_file}' is stale/not found. Rebuilding FAISS index...")
                    embeddings = self._get_embeddings(self.all_messages, "passage")
                    if embeddings.shape[0] > 0:
                        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
                        self.faiss_index.add(embeddings.astype('float32'))
                        print(f"💾 Saving new FAISS index to cache: {FAISS_INDEX_CACHED_PATH}")
                        faiss.write_index(self.faiss_index, FAISS_INDEX_CACHED_PATH)
                        print("✅ FAISS index rebuilt and cached.")
                    else:
                        self.faiss_index = None
                        print("🔴 WARNING: No data to build FAISS index for k-NN.")
        else:
            self.faiss_index = None; self.all_messages = []; self.all_labels = []
            print("INFO: Skipping FAISS index build (mode is NB only).")
            
        self.is_loaded = True
        print(f"--- SpamGuard AI Classifier is now fully loaded in '{mode}' mode. ---")

    def _ensure_loaded(self):
        """Helper method to check if the models are loaded before use."""
        if not self.is_loaded:
            self.load()

    def reload(self):
        """Triggers a full reload of the classifier components on the next request."""
        print("--- Reload triggered. Models will be reloaded on the next request. ---")
        self.is_loaded = False

    def _average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def _get_embeddings(self, texts: list, prefix: str, batch_size: int = 32) -> np.ndarray:
        if not texts or self.transformer_model is None:
            return np.array([], dtype=np.float32).reshape(0, 768)
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [f"{prefix}: {text}" for text in texts[i:i + batch_size]]
            batch_dict = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.transformer_model(**batch_dict)
            embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)
    
    def get_nb_probabilities(self, messages: List[str]):
        self._ensure_loaded()
        if not self.nb_pipeline:
            return {"error": "Naive Bayes model is not loaded for the current mode."}
        spam_idx = np.where(self.label_encoder.classes_ == 'spam')[0][0]
        all_probs = self.nb_pipeline.predict_proba(messages)
        return {"spam_probabilities": [p[spam_idx] for p in all_probs]}

    def explain_model(self, top_n: int = 20):
        self._ensure_loaded()
        if not self.nb_pipeline:
            return {"error": "Naive Bayes model is not loaded for the current mode."}
        try:
            vectorizer = self.nb_pipeline.named_steps['tfidf']
            model = self.nb_pipeline.named_steps['clf']
            feature_names = np.array(vectorizer.get_feature_names_out())
            log_probs = model.feature_log_prob_
            spam_idx = np.where(self.label_encoder.classes_ == 'spam')[0][0]
            ham_idx = np.where(self.label_encoder.classes_ == 'ham')[0][0]
            top_spam_indices = log_probs[spam_idx].argsort()[-top_n:][::-1]
            top_ham_indices = log_probs[ham_idx].argsort()[-top_n:][::-1]
            return {
                "top_spam_keywords": feature_names[top_spam_indices].tolist(),
                "top_ham_keywords": feature_names[top_ham_indices].tolist()
            }
        except Exception as e:
            return {"error": f"An error occurred during explanation: {e}"}

    def classify(self, text: str) -> dict:
        self._ensure_loaded()
        current_config = registry.get_current_config()
        mode = current_config.get("mode", "hybrid")

        # --- Stage 1: Naive Bayes Logic ---
        if self.nb_pipeline and self.label_encoder and mode in ["hybrid", "nb_only"]:
            nb_probabilities = self.nb_pipeline.predict_proba([text])[0]
            if mode == "nb_only":
                pred_idx = np.argmax(nb_probabilities)
                pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
                return {"prediction": pred_label, "confidence": max(nb_probabilities), "model": "MultinomialNB (Only)", "evidence": None}
            
            spam_prob = nb_probabilities[np.where(self.label_encoder.classes_ == 'spam')[0][0]]
            if spam_prob < 0.15:
                return {"prediction": "ham", "confidence": 1 - spam_prob, "model": "MultinomialNB", "evidence": None}
            if spam_prob > 0.85:
                return {"prediction": "spam", "confidence": spam_prob, "model": "MultinomialNB", "evidence": None}

        # --- Stage 2: k-NN Logic ---
        if self.faiss_index and self.faiss_index.ntotal > 0 and mode in ["hybrid", "knn_only"]:
            k = 5
            query_embedding = self._get_embeddings([text], "query", batch_size=1)
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            neighbor_labels = [self.all_labels[i] for i in indices[0]]
            prediction = max(set(neighbor_labels), key=neighbor_labels.count)
            confidence = neighbor_labels.count(prediction) / k
            evidence = [{"similar_message": self.all_messages[idx], "label": self.all_labels[idx], "similarity_score": float(scores[0][i])} for i, idx in enumerate(indices[0])]
            return {"prediction": prediction, "confidence": confidence, "model": "Vector Search (k-NN)", "evidence": evidence}
        
        # --- Fallback if the designated model for the mode is unavailable ---
        print(f"🔴 WARNING: No model/index loaded for current mode ('{mode}'). Falling back.")
        return {"prediction": "ham", "confidence": 0.5, "model": "Fallback", "evidence": None}