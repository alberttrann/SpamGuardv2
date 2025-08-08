# evaluate_all_architectures.py 

import os
import sys
import pandas as pd
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import time
import joblib

# --- Robust Pathing and Imports ---
PROJECT_ROOT_PATH = os.getcwd()
if PROJECT_ROOT_PATH not in sys.path: sys.path.append(PROJECT_ROOT_PATH)
from backend.utils import preprocess_tokenizer
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --- Configuration ---
ORIGINAL_DATA_PATH = r"C:\Users\alberttran\Downloads\2cls_spam_text_cls_original.csv"
CURRENT_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'backend', 'data', '2cls_spam_text_cls.csv')
EVALUATION_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "evaluation_data.txt")

# --- Global Components  ---
print("Loading sentence-transformer model (intfloat/multilingual-e5-base)...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
TRANSFORMER_MODEL = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(DEVICE).eval()
print(f"Model loaded on {DEVICE}.")

# --- Helper Functions for Vectorization ---
def average_pool(states, mask): return (states * mask[..., None]).sum(1) / mask.sum(-1)[..., None]
def get_embeddings(texts: list, prefix: str, batch_size: int = 32) -> np.ndarray:
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = [f"{prefix}: {text}" for text in texts[i:i + batch_size]]
        tokens = TOKENIZER(batch, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): outputs = TRANSFORMER_MODEL(**tokens)
        embeds = average_pool(outputs.last_hidden_state, tokens['attention_mask'])
        all_embeds.append(F.normalize(embeds, p=2, dim=1).cpu().numpy())
    return np.vstack(all_embeds)

# --- Classifier Definitions ---

class Classifier_kNN_Only:
    def __init__(self, db_messages, db_labels, faiss_index, k=5):
        self.db_messages = db_messages; self.db_labels = db_labels
        self.faiss_index = faiss_index; self.k = k
    def classify(self, message):
        start_time = time.perf_counter()
        query_embedding = get_embeddings([message], "query", 1)
        _, indices = self.faiss_index.search(query_embedding.astype('float32'), self.k)
        neighbor_labels = [self.db_labels[i] for i in indices[0]]
        prediction = max(set(neighbor_labels), key=neighbor_labels.count)
        confidence = neighbor_labels.count(prediction) / self.k
        end_time = time.perf_counter()
        return {"prediction": prediction, "confidence": confidence, "model": "k-NN", "time_ms": (end_time - start_time) * 1000}

class Classifier_Multinomial_Only:
    def __init__(self, pipeline, label_encoder):
        self.pipeline = pipeline; self.label_encoder = label_encoder
    def classify(self, message):
        start_time = time.perf_counter()
        numeric_pred = self.pipeline.predict([message])[0]
        prediction = self.label_encoder.inverse_transform([numeric_pred])[0]
        probs = self.pipeline.predict_proba([message])[0]
        confidence = max(probs)
        end_time = time.perf_counter()
        return {"prediction": prediction, "confidence": confidence, "model": "MultinomialNB", "time_ms": (end_time - start_time) * 1000}

class Classifier_Hybrid:
    def __init__(self, pipeline, label_encoder, db_messages, db_labels, faiss_index, k=5):
        self.nb_classifier = Classifier_Multinomial_Only(pipeline, label_encoder)
        self.knn_classifier = Classifier_kNN_Only(db_messages, db_labels, faiss_index, k)
        self.label_encoder = label_encoder
    def classify(self, message):
        start_time = time.perf_counter()
        # Stage 1: Fast Triage with Naive Bayes
        nb_probs = self.nb_classifier.pipeline.predict_proba([message])[0]
        spam_class_index = np.where(self.label_encoder.classes_ == 'spam')[0][0]
        spam_prob = nb_probs[spam_class_index]

        if spam_prob < 0.15:
            prediction = "ham"; confidence = 1 - spam_prob; model_used = "MultinomialNB"
            end_time = time.perf_counter()
        elif spam_prob > 0.85:
            prediction = "spam"; confidence = spam_prob; model_used = "MultinomialNB"
            end_time = time.perf_counter()
        else:
            # Stage 2: Deep Analysis with k-NN (re-measure time for just this part)
            knn_result = self.knn_classifier.classify(message)
            prediction = knn_result["prediction"]; confidence = knn_result["confidence"]
            model_used = "Vector Search (k-NN)"; end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        return {"prediction": prediction, "confidence": confidence, "model": model_used, "time_ms": total_time_ms}


def run_full_evaluation(dataset_path, dataset_name):
    """
    Main function to build all classifiers from a dataset and evaluate them.
    """
    print("\n" + "="*80)
    print(f"  STARTING FULL EVALUATION SUITE ON: {dataset_name.upper()}")
    print("="*80)

    # --- 1. Load and Prepare Data ---
    print(f"\n[Step 1/4] Loading and preparing data from '{os.path.basename(dataset_path)}'...")
    if not os.path.exists(dataset_path): print(f"❌ ERROR: File not found."); return
    df = pd.read_csv(dataset_path, quotechar='"', on_bad_lines='skip')
    df.dropna(subset=['Message', 'Category'], inplace=True); df.drop_duplicates(subset=['Message'], inplace=True)
    
    X_train = df["Message"].astype(str)
    y_train_labels = df["Category"]
    le = LabelEncoder(); y_train = le.fit_transform(y_train_labels)

    with open(EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f: lines = f.readlines()
    true_labels = []; eval_messages = []
    for line in lines:
        if line.strip(): label, msg = line.split(',', 1); true_labels.append(label.strip()); eval_messages.append(msg.strip())
    
    # --- 2. Build All Classifiers ---
    print("\n[Step 2/4] Building all classifier architectures...")
    
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=preprocess_tokenizer, stop_words=None, ngram_range=(1, 2), max_features=10000)),
        ('smote', SMOTE(random_state=42)),
        ('clf', MultinomialNB(alpha=0.1))
    ])
    nb_pipeline.fit(X_train, y_train)
    
    db_messages = df["Message"].astype(str).tolist(); db_labels = df["Category"].tolist()
    passage_embeddings = get_embeddings(db_messages, "passage")
    faiss_index = faiss.IndexFlatIP(passage_embeddings.shape[1]); faiss_index.add(passage_embeddings.astype('float32'))
    print("✅ All models and indexes are built.")

    classifiers = {
        "MultinomialNB Only": Classifier_Multinomial_Only(nb_pipeline, le),
        "k-NN Only": Classifier_kNN_Only(db_messages, db_labels, faiss_index),
        "Hybrid System": Classifier_Hybrid(nb_pipeline, le, db_messages, db_labels, faiss_index)
    }

    # --- 3. Run and Time Evaluations ---
    print("\n[Step 3/4] Running evaluations for each architecture...")
    all_results = {}
    for name, classifier in classifiers.items():
        print(f"\n--- Testing: {name} ---")
        predictions = []; total_time = 0
        for msg in tqdm(eval_messages, desc=f"Classifying with {name}"):
            result = classifier.classify(msg)
            predictions.append(result)
            total_time += result['time_ms']
        all_results[name] = {"predictions": predictions, "total_time_ms": total_time}

    # --- 4. Print Detailed Reports ---
    print("\n[Step 4/4] Generating detailed performance and timing reports...")
    for name, results_data in all_results.items():
        print("\n" + "-"*60)
        print(f"  DETAILED REPORT FOR: {name} on {dataset_name}")
        print("-"*60)
        
        preds = results_data["predictions"]
        pred_labels = [p['prediction'] for p in preds]
        
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"Overall Accuracy: {accuracy:.2%}")
        
        total_time_s = results_data['total_time_ms'] / 1000
        avg_time_ms = results_data['total_time_ms'] / len(preds)
        print(f"Total Prediction Time: {total_time_s:.4f} seconds")
        print(f"Average Prediction Time: {avg_time_ms:.4f} ms/message")

        if name == "Hybrid System":
            nb_count = sum(1 for p in preds if p['model'] == 'MultinomialNB')
            knn_count = sum(1 for p in preds if p['model'] == 'Vector Search (k-NN)')
            nb_correct = sum(1 for p, true in zip(preds, true_labels) if p['model'] == 'MultinomialNB' and p['prediction'] == true)
            knn_correct = sum(1 for p, true in zip(preds, true_labels) if p['model'] == 'Vector Search (k-NN)' and p['prediction'] == true)
            
            print("\nHybrid Model Usage:")
            print(f"  - MultinomialNB used: {nb_count} times ({nb_count/len(preds):.1%})")
            print(f"  - Vector Search (k-NN) used: {knn_count} times ({knn_count/len(preds):.1%})")
            
            nb_acc = (nb_correct / nb_count) if nb_count > 0 else 0
            knn_acc = (knn_correct / knn_count) if knn_count > 0 else 0
            print(f"  - Accuracy of NB Triage: {nb_acc:.2%}")
            print(f"  - Accuracy of k-NN Escalation: {knn_acc:.2%}")
        
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, target_names=le.classes_))


if __name__ == "__main__":
    run_full_evaluation(ORIGINAL_DATA_PATH, "Original Biased Dataset")
    run_full_evaluation(CURRENT_DATA_PATH, "Current Augmented Dataset")
    print("\n" + "="*80)
    print("  ALL EXPERIMENTS COMPLETE")
    print("="*80)