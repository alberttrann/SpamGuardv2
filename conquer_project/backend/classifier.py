import joblib
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import os

class SpamGuardClassifier:
    """
    A hybrid classifier that uses Naive Bayes for fast triage and a powerful
    Transformer-based vector search for deep analysis. It is designed to be
    reloaded on-the-fly after retraining.
    """
    def __init__(self, model_path_dir="E:\AIO\conquer_project\models"):
        """Initializes the classifier by loading all necessary components."""
        # The main loading logic is separated to be reusable for reloading.
        self._load_all_components(model_path_dir)

    def _load_all_components(self, model_path_dir="E:\AIO\conquer_project\models"):
        """
        Loads all models, data, and builds the FAISS index.
        This method is called on init and during a manual reload.
        """
        print("--- Initializing or Reloading SpamGuard AI Classifier ---")
        
        # --- 1. Load Naive Bayes Components ---
        self.nb_model = joblib.load(os.path.join(model_path_dir, 'nb_model.joblib'))
        self.dictionary = joblib.load(os.path.join(model_path_dir, 'dictionary.joblib'))
        self.label_encoder = joblib.load(os.path.join(model_path_dir, 'label_encoder.joblib'))
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        print("✅ Naive Bayes components loaded.")

        # --- 2. Load Transformer Model (Optimization: only load if it doesn't exist) ---
        if not hasattr(self, 'transformer_model'):
            MODEL_NAME = "intfloat/multilingual-e5-base"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.transformer_model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()
            print(f"✅ Transformer model loaded for the first time on {self.device}.")

        # --- 3. Load Fresh Data and Rebuild FAISS Index ---
        df = pd.read_csv(
            "data/2cls_spam_text_cls.csv", 
            quotechar='"', 
            on_bad_lines='skip'
        )
        # Clean data to prevent errors from bad LLM generations or empty rows
        df.dropna(subset=['Message'], inplace=True)
        self.all_messages = df["Message"].astype(str).tolist()
        self.all_labels = df["Category"].tolist()
        
        print("Generating embeddings for the entire dataset to build FAISS index...")
        embeddings = self._get_embeddings(self.all_messages, "passage")
        self.embedding_dim = embeddings.shape[1]
        
        # We rebuild the index from scratch with the potentially updated dataset
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings.astype('float32'))
        print("✅ FAISS index built/rebuilt successfully.")
        print("--- SpamGuard AI Classifier is ready. ---")

    def reload(self):
        """
        Public method to trigger a full reload of the Naive Bayes model
        and a rebuild of the FAISS index from the updated dataset.
        """
        self._load_all_components()

    def _preprocess_nb(self, text: str) -> list:
        """Preprocessing steps specific to the Naive Bayes model."""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def _create_features_nb(self, tokens: list) -> np.ndarray:
        """Creates a Bag-of-Words feature vector for Naive Bayes."""
        features = np.zeros(len(self.dictionary))
        for token in tokens:
            if token in self.dictionary:
                features[self.dictionary.index(token)] += 1
        return features

    def _average_pool(self, last_hidden_states, attention_mask):
        """Pools the transformer outputs to get a single sentence embedding."""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def _get_embeddings(self, texts: list, prefix: str, batch_size: int = 32) -> np.ndarray:
        """Generates sentence embeddings for a list of texts in memory-efficient batches."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
            batch_texts = texts[i:i + batch_size]
            prefixed_batch = [f"{prefix}: {text}" for text in batch_texts]
            
            batch_dict = self.tokenizer(prefixed_batch, max_length=512, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.transformer_model(**batch_dict)
            
            embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)

    def classify(self, text: str) -> dict:
        """
        Classifies a single text message using the hybrid approach.
        """
        # --- Stage 1: Fast Triage with Naive Bayes ---
        processed_text_nb = self._preprocess_nb(text)
        features_nb = self._create_features_nb(processed_text_nb)
        features_nb = np.array(features_nb).reshape(1, -1)
        
        nb_probabilities = self.nb_model.predict_proba(features_nb)[0]
        # Find the index for 'spam' to get its probability
        spam_class_index = np.where(self.label_encoder.classes_ == 'spam')[0][0]
        spam_prob = nb_probabilities[spam_class_index]

        # Triage thresholds: if NB is very confident, we trust it.
        # Note: These thresholds can be tuned for performance.
        if spam_prob < 0.1:
            return {"prediction": "ham", "confidence": 1 - spam_prob, "model": "Naive Bayes", "evidence": None}
        if spam_prob > 0.9:
            return {"prediction": "spam", "confidence": spam_prob, "model": "Naive Bayes", "evidence": None}

        # --- Stage 2: Deep Analysis with Vector Search for ambiguous cases ---
        k = 5 # Number of nearest neighbors to consider
        query_embedding = self._get_embeddings([text], "query").astype('float32') # Process as a list
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        neighbor_labels = [self.all_labels[i] for i in indices[0]]
        
        # Majority Vote determines the final prediction
        prediction = max(set(neighbor_labels), key=neighbor_labels.count)
        
        # Confidence is the ratio of the winning class in the neighbors
        confidence = neighbor_labels.count(prediction) / k

        # Gather evidence for explainability
        evidence = []
        for i, idx in enumerate(indices[0]):
            evidence.append({
                "similar_message": self.all_messages[idx],
                "label": self.all_labels[idx],
                "similarity_score": float(scores[0][i])
            })

        return {
            "prediction": prediction, 
            "confidence": confidence, 
            "model": "Vector Search (k-NN)", 
            "evidence": evidence
        }