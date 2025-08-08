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

# Use relative imports for local modules
from .utils import preprocess_tokenizer
from . import registry

# --- Define robust absolute paths ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, '..'))
DATA_CSV_PATH = os.path.join(BACKEND_DIR, 'data', '2cls_spam_text_cls.csv')


class SpamGuardClassifier:
    """
    A hybrid classifier that uses a superior Multinomial Naive Bayes pipeline for
    fast triage and a powerful Transformer-based vector search for deep analysis.
    It is designed to be reloaded on-the-fly after retraining.
    """
    def __init__(self):
        """Initializes the classifier by loading the active model from the registry."""
        self._load_all_components()

    def _load_all_components(self):
        """
        Loads the ACTIVE model from the registry, data, and builds the FAISS index.
        """
        print("--- Initializing or Reloading SpamGuard AI Classifier ---")
        
        # --- 1. Load ACTIVE Naive Bayes Pipeline from Registry ---
        pipeline_path, encoder_path = registry.get_active_model_paths()
        
        if pipeline_path and encoder_path and os.path.exists(pipeline_path):
            try:
                self.nb_pipeline = joblib.load(pipeline_path)
                self.label_encoder = joblib.load(encoder_path)
                active_id = registry.get_all_models().get("active_model_id", "N/A")
                print(f"✅ Active model '{active_id}' loaded successfully.")
            except Exception as e:
                print(f"🔴 ERROR loading active model: {e}. Classifier will rely on Vector Search.")
                self.nb_pipeline = None; self.label_encoder = None
        else:
            print(f"🔴 WARNING: No active Naive Bayes model found in registry.")
            print("Please run `python -m backend.train_nb` to create an initial model.")
            self.nb_pipeline = None; self.label_encoder = None

        # --- 2. Load Transformer Model ---
        if not hasattr(self, 'transformer_model'):
            MODEL_NAME = "intfloat/multilingual-e5-base"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.transformer_model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()
            print(f"✅ Transformer model loaded on {self.device}.")

        # --- 3. Load App Data and Build FAISS Index ---
        if not os.path.exists(DATA_CSV_PATH):
            print(f"🔴 WARNING: Data file not found at {DATA_CSV_PATH}. FAISS index will be empty.")
            self.all_messages = []; self.all_labels = []; self.faiss_index = None
        else:
            df = pd.read_csv(DATA_CSV_PATH, quotechar='"', on_bad_lines='skip')
            df.dropna(subset=['Message'], inplace=True)
            self.all_messages = df["Message"].astype(str).tolist()
            self.all_labels = df["Category"].tolist()
            
            print("Generating embeddings for the entire dataset to build FAISS index...")
            embeddings = self._get_embeddings(self.all_messages, "passage")
            
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_index.add(embeddings.astype('float32'))
            print("✅ FAISS index built/rebuilt successfully.")
            
        print("--- SpamGuard AI Classifier is ready. ---")

    def reload(self):
        """Triggers a full reload of the classifier and FAISS index."""
        print("--- Reloading classifier with updated data... ---")
        self._load_all_components()

    def _average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def _get_embeddings(self, texts: list, prefix: str, batch_size: int = 32) -> np.ndarray:
        if not texts:
            # Handle case where there's no data to embed
            return np.array([])
            
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

    def classify(self, text: str) -> dict:
        """Classifies a single text message using the hybrid approach."""
        # --- Stage 1: Fast Triage with MultinomialNB ---
        if self.nb_pipeline and self.label_encoder:
            nb_probabilities = self.nb_pipeline.predict_proba([text])[0]
            spam_class_index = np.where(self.label_encoder.classes_ == 'spam')[0][0]
            spam_prob = nb_probabilities[spam_class_index]

            if spam_prob < 0.15:
                return {"prediction": "ham", "confidence": 1 - spam_prob, "model": "MultinomialNB", "evidence": None}
            if spam_prob > 0.85:
                return {"prediction": "spam", "confidence": spam_prob, "model": "MultinomialNB", "evidence": None}

        # --- Stage 2: Deep Analysis with Vector Search ---
        # This stage is triggered if the NB model is uncertain OR if it failed to load/FAISS is empty.
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            # Fallback if FAISS is not available
            return {"prediction": "ham", "confidence": 0.5, "model": "Fallback (No FAISS Index)", "evidence": None}

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