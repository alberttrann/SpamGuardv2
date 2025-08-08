# evaluate_hybrid_before_after.py

import os
import sys
import pandas as pd
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

# --- Robust Pathing and Imports ---
PROJECT_ROOT_PATH = os.getcwd()
if PROJECT_ROOT_PATH not in sys.path:
    sys.path.append(PROJECT_ROOT_PATH)
from backend.utils import preprocess_tokenizer

# --- Configuration: Define all file and model paths ---
BACKEND_DIR = os.path.join(PROJECT_ROOT_PATH, 'backend')
DATA_DIR = os.path.join(BACKEND_DIR, 'data')

# Training data sources
DATA_PATH_BEFORE = os.path.join(DATA_DIR, 'before_270.csv')
DATA_PATH_AFTER = os.path.join(DATA_DIR, '2cls_spam_text_cls.csv')

# Model IDs
MODELS_DIR = os.path.join(PROJECT_ROOT_PATH, 'models')
MODEL_ID_BEFORE = "nb_multinomial_20250803_222542"
MODEL_ID_AFTER = "nb_multinomial_20250804_075009"

# Test sets
MIXED_TEST_SET_PATH = os.path.join(PROJECT_ROOT_PATH, 'mixed_test_set.txt')
TRICKY_HAM_TEST_SET_PATH = os.path.join(PROJECT_ROOT_PATH, 'only_tricky_ham_test_set.txt')

# --- Global Components (Load Once) ---
print("Loading sentence-transformer model (intfloat/multilingual-e-base)...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
TRANSFORMER_MODEL = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(DEVICE).eval()
print(f"Model loaded on {DEVICE}.")

# --- Helper Functions ---
def average_pool(states, mask): return (states * mask[..., None]).sum(1) / mask.sum(-1)[..., None]
def get_embeddings(texts: list, prefix: str, batch_size: int = 32) -> np.ndarray:
    all_embeds = []
    # No tqdm here for cleaner final report
    for i in range(0, len(texts), batch_size):
        batch = [f"{prefix}: {text}" for text in texts[i:i + batch_size]]
        tokens = TOKENIZER(batch, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): outputs = TRANSFORMER_MODEL(**tokens)
        embeds = average_pool(outputs.last_hidden_state, tokens['attention_mask'])
        all_embeds.append(F.normalize(embeds, p=2, dim=1).cpu().numpy())
    return np.vstack(all_embeds)

def load_test_set(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    true_labels, eval_messages = [], []
    for line in lines:
        if line.strip():
            parts = line.split(',', 1)
            if len(parts) == 2 and parts[0].strip().lower() in ['ham', 'spam']:
                true_labels.append(parts[0].strip().lower())
                eval_messages.append(parts[1].strip())
    return true_labels, eval_messages

def print_report(results, true_labels, model_name, test_set_name):
    """Prints a standardized evaluation report from the collected results."""
    print("\n" + "="*60)
    print(f"  RESULTS FOR: {model_name}")
    print(f"  ON TEST SET: {test_set_name}")
    print("="*60)

    pred_labels = [r['prediction'] for r in results]
    total_time_ms = sum(r['time_ms'] for r in results)

    accuracy = accuracy_score(true_labels, pred_labels)
    avg_time_ms = total_time_ms / len(results)

    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Total Prediction Time: {total_time_ms / 1000:.4f} seconds")
    print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")

    # Hybrid specific stats
    nb_count = sum(1 for r in results if r['model'] == 'MultinomialNB')
    knn_count = sum(1 for r in results if r['model'] == 'Vector Search (k-NN)')
    if nb_count + knn_count == len(results): # Check if it's a hybrid run
        nb_correct = sum(1 for r, true in zip(results, true_labels) if r['model'] == 'MultinomialNB' and r['prediction'] == true)
        knn_correct = sum(1 for r, true in zip(results, true_labels) if r['model'] == 'Vector Search (k-NN)' and r['prediction'] == true)
        nb_acc = (nb_correct / nb_count) if nb_count > 0 else 0
        knn_acc = (knn_correct / knn_count) if knn_count > 0 else 0
        
        print("\nHybrid Model Usage:")
        print(f"  - MultinomialNB used: {nb_count} times ({nb_count/len(results):.1%})")
        print(f"  - Vector Search (k-NN) used: {knn_count} times ({knn_count/len(results):.1%})")
        print(f"  - Accuracy of NB Triage: {nb_acc:.2%}")
        print(f"  - Accuracy of k-NN Escalation: {knn_acc:.2%}")

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, labels=['ham', 'spam'], zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels, labels=['ham', 'spam'])
    print(cm)


def run_hybrid_evaluation(model_id: str, model_version_name: str, training_csv_path: str, test_set_path: str):
    """
    Main function to build and evaluate a specific version of the hybrid classifier.
    """
    test_set_name = os.path.basename(test_set_path)
    model_name = f"Hybrid System ({model_version_name})"
    print(f"\n--- Preparing to evaluate: {model_name} on {test_set_name} ---")

    # 1. Load the specific Naive Bayes model version
    pipeline_path = os.path.join(MODELS_DIR, f"{model_id}_pipeline.joblib")
    encoder_path = os.path.join(MODELS_DIR, f"{model_id}_encoder.joblib")
    try:
        nb_pipeline = joblib.load(pipeline_path); le = joblib.load(encoder_path)
    except FileNotFoundError:
        print(f"❌ ERROR: Model files for ID '{model_id}' not found. Skipping."); return

    # 2. Build the FAISS index from the corresponding training data
    df = pd.read_csv(training_csv_path, quotechar='"', on_bad_lines='skip')
    df.dropna(subset=['Message', 'Category'], inplace=True)
    db_messages = df["Message"].astype(str).tolist(); db_labels = df["Category"].tolist()
    passage_embeddings = get_embeddings(db_messages, "passage")
    faiss_index = faiss.IndexFlatIP(passage_embeddings.shape[1]); faiss_index.add(passage_embeddings.astype('float32'))
    
    # 3. Load the test data
    true_labels, eval_messages = load_test_set(test_set_path)
    
    # 4. Run the hybrid classification logic
    results = []
    print(f"Classifying {len(eval_messages)} messages...")
    for message in tqdm(eval_messages, desc=f"Testing {model_version_name}"):
        start_time = time.perf_counter()
        
        # Stage 1: Naive Bayes Triage
        nb_probs = nb_pipeline.predict_proba([message])[0]
        spam_idx = np.where(le.classes_ == 'spam')[0][0]
        spam_prob = nb_probs[spam_idx]
        
        if spam_prob < 0.15:
            prediction = "ham"; confidence = 1 - spam_prob; model_used = "MultinomialNB"
        elif spam_prob > 0.85:
            prediction = "spam"; confidence = spam_prob; model_used = "MultinomialNB"
        else:
            # Stage 2: k-NN Escalation
            model_used = "Vector Search (k-NN)"
            q_emb = get_embeddings([message], "query", 1)
            _, indices = faiss_index.search(q_emb.astype('float32'), 5)
            n_labels = [db_labels[i] for i in indices[0]]
            prediction = max(set(n_labels), key=n_labels.count)
            confidence = n_labels.count(prediction) / 5

        end_time = time.perf_counter()
        results.append({
            "prediction": prediction, 
            "confidence": confidence, 
            "model": model_used, 
            "time_ms": (end_time - start_time) * 1000
        })

    # 5. Print the final, detailed report
    print_report(results, true_labels, model_name, test_set_name)


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("###   EVALUATING MODEL 'BEFORE' (ID: ...222542)   ###")
    print("#"*80)
    run_hybrid_evaluation(MODEL_ID_BEFORE, "Before Retraining", DATA_PATH_BEFORE, MIXED_TEST_SET_PATH)
    run_hybrid_evaluation(MODEL_ID_BEFORE, "Before Retraining", DATA_PATH_BEFORE, TRICKY_HAM_TEST_SET_PATH)

    print("\n" + "#"*80)
    print("###   EVALUATING MODEL 'AFTER' (ID: ...075009)   ###")
    print("#"*80)
    run_hybrid_evaluation(MODEL_ID_AFTER, "After Retraining", DATA_PATH_AFTER, MIXED_TEST_SET_PATH)
    run_hybrid_evaluation(MODEL_ID_AFTER, "After Retraining", DATA_PATH_AFTER, TRICKY_HAM_TEST_SET_PATH)
    
    print("\n" + "="*80)
    print("  ALL HYBRID SYSTEM EVALUATIONS COMPLETE")
    print("="*80)