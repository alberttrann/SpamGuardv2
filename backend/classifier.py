# backend/classifier.py

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
DATA_CSV_PATH = os.path.join(BACKEND_DIR, 'data', '2cls_spam_text_cls.csv')
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, 'models', 'faiss_index.bin')



class SpamGuardClassifier:
    """
    A hybrid classifier that uses a superior Multinomial Naive Bayes pipeline for
    fast triage and a powerful Transformer-based vector search for deep analysis.
    This class uses lazy loading to ensure fast server startups.
    """
    def __init__(self):
        """
        Initializes the classifier in a 'lazy' state.
        The models are not loaded until they are explicitly needed.
        """
        print("SpamGuardClassifier instance created in a lazy state.")
        self.is_loaded = False
        # Initialize all model attributes to None
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
        Loads all models and data. It now uses an intelligent caching system
        for the FAISS index to avoid slow re-computation.
        """
        print("--- LAZY LOADING: Initializing or Reloading SpamGuard AI Classifier ---")
        
        # --- 1. Load Naive Bayes Pipeline (Unchanged) ---
        pipeline_path, encoder_path = registry.get_active_model_paths()
        if pipeline_path and encoder_path:
            try:
                self.nb_pipeline = joblib.load(pipeline_path)
                self.label_encoder = joblib.load(encoder_path)
                active_id = registry.get_all_models().get("active_model_id", "N/A")
                print(f"✅ Active model '{active_id}' loaded successfully.")
            except Exception as e:
                print(f"🔴 ERROR loading active model: {e}. NB triage will be skipped.")
                self.nb_pipeline = None; self.label_encoder = None
        else:
            print(f"🔴 WARNING: No active Naive Bayes model found in registry.")
            self.nb_pipeline = None; self.label_encoder = None

        # --- 2. Load Transformer Model (Unchanged) ---
        if self.transformer_model is None:
            MODEL_NAME = "intfloat/multilingual-e5-base"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.transformer_model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()
            print(f"✅ Transformer model loaded on {self.device}.")

        # --- 3. Load App Data and Build/Load FAISS Index (NEW CACHING LOGIC) ---
        if not os.path.exists(DATA_CSV_PATH):
            print(f"🔴 WARNING: Data file not found at {DATA_CSV_PATH}. FAISS index cannot be built.")
            self.all_messages = []; self.all_labels = []; self.faiss_index = None
        else:
            df = pd.read_csv(DATA_CSV_PATH, quotechar='"', on_bad_lines='skip')
            df.dropna(subset=['Message'], inplace=True)
            self.all_messages = df["Message"].astype(str).tolist()
            self.all_labels = df["Category"].tolist()
            
            # --- THIS IS THE NEW CACHING LOGIC ---
            # Check if a cached index exists and if it's newer than the data file.
            cache_is_valid = (
                os.path.exists(FAISS_INDEX_PATH) and
                os.path.getmtime(FAISS_INDEX_PATH) >= os.path.getmtime(DATA_CSV_PATH)
            )
            
            if cache_is_valid:
                print("✅ Found valid cache. Loading FAISS index from disk... (This is fast)")
                self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                print("✅ FAISS index loaded from cache.")
            else:
                print("🔴 Cache is stale or not found. Rebuilding FAISS index... (This is the slow part)")
                embeddings = self._get_embeddings(self.all_messages, "passage")
                
                if embeddings.shape[0] > 0:
                    self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
                    self.faiss_index.add(embeddings.astype('float32'))
                    
                    # Save the newly built index to the cache file
                    print(f"💾 Saving new FAISS index to cache: {FAISS_INDEX_PATH}")
                    faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
                    print("✅ FAISS index rebuilt and cached successfully.")
                else:
                    self.faiss_index = None
                    print("🔴 WARNING: No data to build FAISS index.")
            
        self.is_loaded = True
        print("--- SpamGuard AI Classifier is now fully loaded and ready. ---")

    def _ensure_loaded(self):
        """A helper method to check if the models are loaded before use."""
        if not self.is_loaded:
            self.load()

    def reload(self):
        """Triggers a full reload of the classifier components."""
        # We only need to mark as not loaded. The next call to _ensure_loaded will do the work.
        print("--- Reload triggered. Models will be reloaded on the next request. ---")
        self.is_loaded = False

    def _average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def _get_embeddings(self, texts: list, prefix: str, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 768) # Return empty array with correct shape
            
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
        """
        Ensures model is loaded and returns raw spam probabilities for a list of messages.
        """
        self._ensure_loaded()
        if not self.nb_pipeline:
            return {"error": "Naive Bayes model is not loaded."}
        
        spam_idx = np.where(self.label_encoder.classes_ == 'spam')[0][0]
        all_probs = self.nb_pipeline.predict_proba(messages)
        return {"spam_probabilities": [p[spam_idx] for p in all_probs]}

    def explain_model(self, top_n: int = 20):
        """
        Ensures model is loaded and returns the top keywords.
        """
        self._ensure_loaded()
        if not self.nb_pipeline:
            return {"error": "Naive Bayes model not loaded."}
        
        try:
            vectorizer = self.nb_pipeline.named_steps['tfidf']
            model = self.nb_pipeline.named_steps['clf']
            label_encoder = self.label_encoder
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

    def classify(self, text: str) -> dict:
        """Classifies a single text message using the hybrid approach."""
        self._ensure_loaded() # This check ensures models are loaded before proceeding.
        
        # --- Stage 1: Fast Triage with MultinomialNB ---
        if self.nb_pipeline and self.label_encoder:
            nb_probabilities = self.nb_pipeline.predict_proba([text])[0]
            spam_class_index = np.where(self.label_encoder.classes_ == 'spam')[0][0]
            spam_prob = nb_probabilities[spam_class_index]

            # Stricter threshold based on our findings
            if spam_prob < 0.15: # Confident Ham
                return {"prediction": "ham", "confidence": 1 - spam_prob, "model": "MultinomialNB", "evidence": None}
            if spam_prob > 0.85: # Confident Spam
                return {"prediction": "spam", "confidence": spam_prob, "model": "MultinomialNB", "evidence": None}

        # --- Stage 2: Deep Analysis with Vector Search ---
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            # Fallback if FAISS is not available. Predict based on NB's less confident result.
            if self.nb_pipeline:
                prediction = self.nb_pipeline.predict([text])[0]
                confidence = max(self.nb_pipeline.predict_proba([text])[0])
                return {"prediction": prediction, "confidence": confidence, "model": "MultinomialNB (No FAISS)", "evidence": None}
            else: # Absolute fallback
                return {"prediction": "ham", "confidence": 0.5, "model": "Fallback (No Models)", "evidence": None}

        k = 5
        query_embedding = self._get_embeddings([text], "query", batch_size=1)
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        neighbor_labels = [self.all_labels[i] for i in indices[0]]
        prediction = max(set(neighbor_labels), key=neighbor_labels.count)
        confidence = neighbor_labels.count(prediction) / k
        
        evidence = [
            {
                "similar_message": self.all_messages[idx],
                "label": self.all_labels[idx],
                "similarity_score": float(scores[0][i])
            }
            for i, idx in enumerate(indices[0])
        ]

        return {
            "prediction": prediction, 
            "confidence": confidence, 
            "model": "Vector Search (k-NN)", 
            "evidence": evidence
        }