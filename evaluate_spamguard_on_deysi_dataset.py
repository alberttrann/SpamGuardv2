# evaluate_on_deysi_dataset.py

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
from datasets import load_dataset

PROJECT_ROOT_PATH = os.getcwd()
if PROJECT_ROOT_PATH not in sys.path:
    sys.path.append(PROJECT_ROOT_PATH)
from backend.utils import preprocess_tokenizer

HF_DATASET_ID = "Deysi/spam-detection-dataset"
DATASET_SPLIT = 'test'

MODEL_ID_TO_TEST = "nb_multinomial_20250804_075009"

TRAINING_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'backend', 'data', '2cls_spam_text_cls.csv')

MODELS_DIR = os.path.join(PROJECT_ROOT_PATH, 'models')

# --- Global Components (Load Once) ---
print("Loading sentence-transformer model (intfloat/multilingual-e5-base)...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
TRANSFORMER_MODEL = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(DEVICE).eval()
print(f"Model loaded on {DEVICE}.")

# --- Helper Functions  ---
def average_pool(states, mask): return (states * mask[..., None]).sum(1) / mask.sum(-1)[..., None]
def get_embeddings(texts: list, prefix: str, batch_size: int = 32) -> np.ndarray:
    all_embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({prefix})"):
        batch = [f"{prefix}: {text}" for text in texts[i:i + batch_size]]
        tokens = TOKENIZER(batch, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): outputs = TRANSFORMER_MODEL(**tokens)
        embeds = average_pool(outputs.last_hidden_state, tokens['attention_mask'])
        all_embeds.append(F.normalize(embeds, p=2, dim=1).cpu().numpy())
    return np.vstack(all_embeds)

def evaluate_on_deysi():
    """
    Main function to evaluate the final SpamGuard Hybrid model on the Deysi HF dataset.
    """
    print("\n" + "="*80)
    print(f"  STARTING EVALUATION ON HUGGING FACE DATASET: {HF_DATASET_ID} (Split: {DATASET_SPLIT})")
    print("="*80)

    # --- 1. Load the SpamGuard Hybrid Classifier Components ---
    print(f"\n[Step 1/4] Loading local SpamGuard model '{MODEL_ID_TO_TEST}'...")
    pipeline_path = os.path.join(MODELS_DIR, f"{MODEL_ID_TO_TEST}_pipeline.joblib")
    encoder_path = os.path.join(MODELS_DIR, f"{MODEL_ID_TO_TEST}_encoder.joblib")
    try:
        nb_pipeline = joblib.load(pipeline_path); le = joblib.load(encoder_path)
    except FileNotFoundError:
        print(f"❌ ERROR: Model files for ID '{MODEL_ID_TO_TEST}' not found."); return

    print("\n[Step 2/4] Building FAISS index from local training data...")
    df_train = pd.read_csv(TRAINING_DATA_PATH, quotechar='"', on_bad_lines='skip')
    df_train.dropna(subset=['Message', 'Category'], inplace=True)
    db_messages = df_train["Message"].astype(str).tolist(); db_labels = df_train["Category"].tolist()
    passage_embeddings = get_embeddings(db_messages, "passage")
    faiss_index = faiss.IndexFlatIP(passage_embeddings.shape[1]); faiss_index.add(passage_embeddings.astype('float32'))
    print("✅ SpamGuard model components are ready.")

    # --- 2. Load and Prepare the Hugging Face Dataset ---
    print(f"\n[Step 3/4] Downloading and preparing Hugging Face dataset '{HF_DATASET_ID}'...")
    try:
        dataset = load_dataset(HF_DATASET_ID, split=DATASET_SPLIT)
        # Filter out rows with missing text or labels
        dataset = dataset.filter(lambda x: x['label'] is not None and x['text'] is not None and x['text'].strip() != "")
        
        # Translate the labels from 'not_spam'/'spam' to 'ham'/'spam'
        true_labels = ["ham" if label == "not_spam" else "spam" for label in dataset['label']]
        eval_messages = dataset['text']
        print(f"✅ Loaded and preprocessed {len(eval_messages)} messages from the '{DATASET_SPLIT}' split.")
    except Exception as e:
        print(f"❌ ERROR: Failed to download or process the Hugging Face dataset. Details: {e}"); return
        
    # --- 3. Run the Hybrid Classification Logic ---
    print(f"\n[Step 4/4] Classifying {len(eval_messages)} messages... This may take time.")
    results = []
    spam_idx = np.where(le.classes_ == 'spam')[0][0]
    start_time_total = time.perf_counter()

    for message in tqdm(eval_messages, desc="Classifying Deysi Dataset"):
        # Stage 1: Naive Bayes Triage
        nb_probs = nb_pipeline.predict_proba([message])[0]
        spam_prob = nb_probs[spam_idx]
        
        if spam_prob > 0.85:
            prediction = "spam"
        elif spam_prob < 0.15:
            prediction = "ham"
        else:
            # Stage 2: k-NN Escalation
            q_emb = get_embeddings([message], "query", 1)
            _, indices = faiss_index.search(q_emb.astype('float32'), 5)
            n_labels = [db_labels[i] for i in indices[0]]
            prediction = max(set(n_labels), key=n_labels.count)
        
        results.append(prediction)

    total_time_s = time.perf_counter() - start_time_total
    print("✅ Predictions complete.")

    # --- 4. Report the Final Results ---
    print("\n" + "="*80)
    print(f"  FINAL REPORT: SpamGuard Hybrid on '{HF_DATASET_ID}'")
    print("="*80)

    accuracy = accuracy_score(true_labels, results)
    avg_time_ms = (total_time_s * 1000) / len(true_labels)
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Total Prediction Time: {total_time_s / 60:.2f} minutes ({total_time_s:.2f} seconds)")
    print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")
    
    all_known_labels = ['ham', 'spam']
    print("\nClassification Report:")
    print(classification_report(true_labels, results, labels=all_known_labels, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, results, labels=all_known_labels)
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=all_known_labels, yticklabels=all_known_labels)
    plt.title(f'SpamGuard Hybrid Performance\non {HF_DATASET_ID} ({DATASET_SPLIT} split)')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.show()


if __name__ == "__main__":
    evaluate_on_deysi()