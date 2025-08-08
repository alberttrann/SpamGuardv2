# evaluate_before_after.py

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

PROJECT_ROOT_PATH = os.getcwd()
if PROJECT_ROOT_PATH not in sys.path:
    sys.path.append(PROJECT_ROOT_PATH)
from backend.utils import preprocess_tokenizer

BACKEND_DIR = os.path.join(PROJECT_ROOT_PATH, 'backend')
DATA_DIR = os.path.join(BACKEND_DIR, 'data')
BEFORE_270_CSV_PATH = os.path.join(DATA_DIR, 'before_270.csv')
CURRENT_CSV_PATH = os.path.join(DATA_DIR, '2cls_spam_text_cls.csv')

MODELS_DIR = os.path.join(PROJECT_ROOT_PATH, 'models')
MODEL_ID_BEFORE = "nb_multinomial_20250803_222542"
MODEL_ID_AFTER = "nb_multinomial_20250804_075009"

MIXED_TEST_SET_PATH = os.path.join(PROJECT_ROOT_PATH, 'mixed_test_set.txt')
TRICKY_HAM_TEST_SET_PATH = os.path.join(PROJECT_ROOT_PATH, 'only_tricky_ham_test_set.txt')

print("Loading sentence-transformer model (intfloat/multilingual-e5-base)...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
TRANSFORMER_MODEL = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(DEVICE).eval()
print(f"Model loaded on {DEVICE}.")

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

def load_test_set(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    true_labels, eval_messages = [], []
    for line in lines:
        if line.strip():
            parts = line.split(',', 1)
            if len(parts) == 2:
                label, message = parts
                if label.strip().lower() in ['ham', 'spam']:
                    true_labels.append(label.strip().lower()); eval_messages.append(message.strip())
    return true_labels, eval_messages

def print_report(predictions, true_labels, model_name, test_set_name, total_time):
    """
    Prints a standardized evaluation report.
    *** MODIFIED to handle NumPy array truthiness correctly. ***
    """
    print("\n" + "="*60)
    print(f"  RESULTS FOR: {model_name}")
    print(f"  ON TEST SET: {test_set_name}")
    print("="*60)
    
    if len(predictions) > 0:
        accuracy = accuracy_score(true_labels, predictions)
        avg_time_ms = (total_time / len(predictions)) * 1000
    else:
        accuracy = 0.0
        avg_time_ms = 0.0

    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")
    
    unique_labels = sorted(list(set(true_labels)))
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, labels=['ham', 'spam'], zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions, labels=['ham', 'spam'])
    print(cm)


def evaluate_knn_model(training_data_path: str, training_data_name: str, test_set_path: str, test_set_name: str):
    model_name = f"k-NN Only (Trained on {training_data_name})"
    print(f"\n--- Preparing: {model_name} ---")
    df = pd.read_csv(training_data_path, quotechar='"', on_bad_lines='skip'); df.dropna(subset=['Message', 'Category'], inplace=True)
    db_messages = df["Message"].astype(str).tolist(); db_labels = df["Category"].tolist()
    passage_embeddings = get_embeddings(db_messages, "passage")
    faiss_index = faiss.IndexFlatIP(passage_embeddings.shape[1]); faiss_index.add(passage_embeddings.astype('float32'))
    
    true_labels, eval_messages = load_test_set(test_set_path)
    
    start_time = time.perf_counter()
    query_embeddings = get_embeddings(eval_messages, "query")
    _, indices = faiss_index.search(query_embeddings.astype('float32'), k=5)
    
    predictions = []
    for neighbor_indices in indices:
        neighbor_labels = [db_labels[i] for i in neighbor_indices]
        prediction = max(set(neighbor_labels), key=neighbor_labels.count)
        predictions.append(prediction)
    total_time = time.perf_counter() - start_time

    print_report(predictions, true_labels, model_name, test_set_name, total_time)

def evaluate_mnb_model(model_id: str, model_version_name: str, test_set_path: str, test_set_name: str):
    model_name = f"MultinomialNB Only ({model_version_name})"
    print(f"\n--- Loading: {model_name} (ID: {model_id}) ---")
    pipeline_path = os.path.join(MODELS_DIR, f"{model_id}_pipeline.joblib")
    encoder_path = os.path.join(MODELS_DIR, f"{model_id}_encoder.joblib")
    
    try:
        pipeline = joblib.load(pipeline_path); le = joblib.load(encoder_path)
    except FileNotFoundError:
        print(f"❌ ERROR: Model files for ID '{model_id}' not found in '{MODELS_DIR}'. Skipping.")
        return

    true_labels, eval_messages = load_test_set(test_set_path)
    
    start_time = time.perf_counter()
    numeric_predictions = pipeline.predict(eval_messages)
    string_predictions = le.inverse_transform(numeric_predictions)
    total_time = time.perf_counter() - start_time
    
    print_report(string_predictions, true_labels, model_name, test_set_name, total_time)

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("###   RUNNING EVALUATION ON MIXED TEST SET   ###")
    print("#"*80)
    evaluate_knn_model(BEFORE_270_CSV_PATH, "Before_270 Data", MIXED_TEST_SET_PATH, "mixed_test_set.txt")
    evaluate_knn_model(CURRENT_CSV_PATH, "Current Data (+270)", MIXED_TEST_SET_PATH, "mixed_test_set.txt")
    evaluate_mnb_model(MODEL_ID_BEFORE, "Before_270 Retrain", MIXED_TEST_SET_PATH, "mixed_test_set.txt")
    evaluate_mnb_model(MODEL_ID_AFTER, "After_270 Retrain", MIXED_TEST_SET_PATH, "mixed_test_set.txt")

    print("\n" + "#"*80)
    print("###   RUNNING EVALUATION ON TRICKY HAM TEST SET   ###")
    print("#"*80)
    evaluate_knn_model(BEFORE_270_CSV_PATH, "Before_270 Data", TRICKY_HAM_TEST_SET_PATH, "only_tricky_ham_test_set.txt")
    evaluate_knn_model(CURRENT_CSV_PATH, "Current Data (+270)", TRICKY_HAM_TEST_SET_PATH, "only_tricky_ham_test_set.txt")
    evaluate_mnb_model(MODEL_ID_BEFORE, "Before_270 Retrain", TRICKY_HAM_TEST_SET_PATH, "only_tricky_ham_test_set.txt")
    evaluate_mnb_model(MODEL_ID_AFTER, "After_270 Retrain", TRICKY_HAM_TEST_SET_PATH, "only_tricky_ham_test_set.txt")
    
    print("\n" + "="*80)
    print("  ALL EXPERIMENTS COMPLETE")
    print("="*80)