# evaluate_knn_only.py

import os
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


ORIGINAL_DATA_PATH = r"C:\Users\alberttran\Downloads\2cls_spam_text_cls_original.csv"

# Path to the current, LLM-augmented dataset
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__)) 
if 'backend' not in BACKEND_DIR: BACKEND_DIR = os.path.join(BACKEND_DIR, 'backend')
CURRENT_DATA_PATH = os.path.join(BACKEND_DIR, 'data', '2cls_spam_text_cls.csv')


EVALUATION_DATA_PATH = "evaluation_data.txt"


print("Loading sentence-transformer model (intfloat/multilingual-e5-base)...")
MODEL_NAME = "intfloat/multilingual-e5-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
TRANSFORMER_MODEL = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print(f"Model loaded on {DEVICE}.")

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embeddings(texts: list, prefix: str, batch_size: int = 32) -> np.ndarray:
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({prefix})"):
        batch_texts = [f"{prefix}: {text}" for text in texts[i:i + batch_size]]
        batch_dict = TOKENIZER(batch_texts, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = TRANSFORMER_MODEL(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

def build_knn_index(dataset_path: str):
    """Loads a dataset, creates embeddings, and builds a FAISS index."""
    print(f"\nBuilding FAISS index from dataset: {os.path.basename(dataset_path)}")
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset file not found at '{dataset_path}'.")
        return None, None, None

    df = pd.read_csv(dataset_path, quotechar='"', on_bad_lines='skip')
    df.dropna(subset=['Message', 'Category'], inplace=True)
    all_messages = df["Message"].astype(str).tolist()
    all_labels = df["Category"].tolist()
    
    passage_embeddings = get_embeddings(all_messages, "passage")
    
    embedding_dim = passage_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(passage_embeddings.astype('float32'))
    print("✅ FAISS index built successfully.")
    
    return faiss_index, all_messages, all_labels

def evaluate_knn_model(dataset_path: str, dataset_name: str, k: int = 5):
    """
    Main evaluation function. Builds a k-NN index from the given dataset
    and evaluates it against the standard evaluation file.
    """
    print("-" * 60)
    print(f"--- Evaluating k-NN ONLY Model (k={k}) on: {dataset_name} ---")
    
    # 1. Build the k-NN search engine
    faiss_index, db_messages, db_labels = build_knn_index(dataset_path)
    if faiss_index is None:
        return 

    # 2. Load the evaluation data
    print(f"Loading evaluation data from '{EVALUATION_DATA_PATH}'...")
    if not os.path.exists(EVALUATION_DATA_PATH):
        print(f"❌ ERROR: Evaluation data file not found at '{EVALUATION_DATA_PATH}'.")
        return
        
    with open(EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f: lines = f.readlines()
    
    true_labels, eval_messages = [], []
    for line in lines:
        if line.strip():
            label, message = line.split(',', 1)
            true_labels.append(label.strip())
            eval_messages.append(message.strip())

    # 3. Classify each message using only k-NN search
    print("Classifying evaluation messages using k-NN search...")
    predictions = []
    query_embeddings = get_embeddings(eval_messages, "query")
    
    scores, indices = faiss_index.search(query_embeddings.astype('float32'), k)
    
    for neighbor_indices in indices:
        neighbor_labels = [db_labels[i] for i in neighbor_indices]
        # majority vote
        prediction = max(set(neighbor_labels), key=neighbor_labels.count)
        predictions.append(prediction)
    print("✅ Predictions complete.")

    # 4. Report the results
    print(f"\n--- Evaluation Report for: {dataset_name} ---")
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nOverall Accuracy: {accuracy:.2%}\n")
    
    print("Classification Report:")
    report = classification_report(true_labels, predictions, zero_division=0)
    print(report)
    
    print("Confusion Matrix:")
    cm = confusion_matrix(true_labels, predictions, labels=['ham', 'spam'])
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='cividis', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title(f'k-NN Only Confusion Matrix\n(Trained on {dataset_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show(block=False) 


if __name__ == "__main__":
    evaluate_knn_model(ORIGINAL_DATA_PATH, "Original Biased Dataset")
    evaluate_knn_model(CURRENT_DATA_PATH, "Current Augmented Dataset")
    
    print("\n" + "="*60)
    print("All evaluations complete. Showing plots.")
    plt.show()