# evaluate_hybrid_system.py

import joblib
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

import sys
PROJECT_ROOT_PATH = os.getcwd()
if PROJECT_ROOT_PATH not in sys.path: sys.path.append(PROJECT_ROOT_PATH)
from backend.utils import preprocess_tokenizer 
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm


def evaluate_final_hybrid_system():
    """
    Loads the final production models (NB pipeline and FAISS index)
    and evaluates the complete hybrid system using the cautious triage logic.
    """
    print("--- Starting Evaluation of Final Hybrid System (MultinomialNB + k-NN) ---")

    # --- Configuration ---
    MODELS_DIR = os.path.join(PROJECT_ROOT_PATH, "models")
    DATA_CSV_PATH = os.path.join(PROJECT_ROOT_PATH, 'backend', 'data', '2cls_spam_text_cls.csv')
    TEST_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "evaluation_data.txt")
    
    # --- 1. Load Stage 1 Model (Naive Bayes Pipeline) ---
    print("\n[Step 1/3] Loading Stage 1 (MultinomialNB) pipeline...")
    try:
        pipeline = joblib.load(os.path.join(MODELS_DIR, "nb_multinomial_pipeline.joblib"))
        label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
        print("✅ Naive Bayes pipeline loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Could not load Naive Bayes model from '{MODELS_DIR}'. Details: {e}")
        return

    # --- 2. Load and Build Stage 2 Model (k-NN Vector Search) ---
    print("\n[Step 2/3] Building Stage 2 (k-NN Vector Search) index...")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOKENIZER = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    TRANSFORMER_MODEL = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(DEVICE).eval()
    print(f"✅ Transformer model loaded on {DEVICE}.")

    # embeddings
    def average_pool(states, mask): return (states * mask[..., None]).sum(1) / mask.sum(-1)[..., None]
    def get_embeddings(texts, prefix, batch_size=32):
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = [f"{prefix}: {text}" for text in texts[i:i + batch_size]]
            tokens = TOKENIZER(batch, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad(): outputs = TRANSFORMER_MODEL(**tokens)
            embeds = average_pool(outputs.last_hidden_state, tokens['attention_mask'])
            all_embeds.append(F.normalize(embeds, p=2, dim=1).cpu().numpy())
        return np.vstack(all_embeds)

    # Load data and build index
    df = pd.read_csv(DATA_CSV_PATH, quotechar='"', on_bad_lines='skip')
    df.dropna(subset=['Message'], inplace=True)
    db_messages = df["Message"].astype(str).tolist()
    db_labels = df["Category"].tolist()
    
    passage_embeddings = get_embeddings(db_messages, "passage")
    faiss_index = faiss.IndexFlatIP(passage_embeddings.shape[1])
    faiss_index.add(passage_embeddings.astype('float32'))
    print("✅ FAISS index built successfully.")

    # --- 3. Load Test Data and Run Cautious Triage Evaluation ---
    print(f"\n[Step 3/3] Loading test data and running evaluation...")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f: lines = f.readlines()
    true_labels = []; messages = []
    for line in lines:
        if line.strip(): label, message = line.split(',', 1); true_labels.append(label.strip()); messages.append(message.strip())
    print(f"✅ Loaded {len(messages)} messages for evaluation.")

    # --- Run the evaluation loop ---
    string_predictions = []
    models_used = []
    for msg in tqdm(messages, desc="Classifying with Hybrid System"):
        # Stage 1: Fast Triage
        nb_probs = pipeline.predict_proba([msg])[0]
        spam_class_index = np.where(label_encoder.classes_ == 'spam')[0][0]
        spam_prob = nb_probs[spam_class_index]

        if spam_prob < 0.15:
            string_predictions.append("ham")
            models_used.append("MultinomialNB")
        elif spam_prob > 0.85:
            string_predictions.append("spam")
            models_used.append("MultinomialNB")
        else:
            # Stage 2: Deep Analysis
            models_used.append("Vector Search (k-NN)")
            query_embedding = get_embeddings([msg], "query", 1)
            _, indices = faiss_index.search(query_embedding.astype('float32'), k=5)
            neighbor_labels = [db_labels[i] for i in indices[0]]
            prediction = max(set(neighbor_labels), key=neighbor_labels.count)
            string_predictions.append(prediction)

    # --- 4. Calculate and Display Metrics ---
    print("\n--- Final Hybrid System Evaluation Results ---")
    
    accuracy = accuracy_score(true_labels, string_predictions)
    print(f"\nOverall Accuracy: {accuracy:.2%}")

    nb_count = models_used.count("MultinomialNB")
    knn_count = models_used.count("Vector Search (k-NN)")
    print("\nHybrid Model Usage:")
    print(f"  - MultinomialNB used: {nb_count} times ({nb_count/len(messages):.1%})")
    print(f"  - Vector Search (k-NN) used: {knn_count} times ({knn_count/len(messages):.1%})")

    print("\nClassification Report:")
    report = classification_report(true_labels, string_predictions, target_names=label_encoder.classes_)
    print(report)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, string_predictions, labels=label_encoder.classes_)
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix for Final Hybrid System')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    evaluate_final_hybrid_system()