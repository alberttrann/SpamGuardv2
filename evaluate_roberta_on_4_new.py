
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import load_dataset

MODEL_ID = "mshenoda/roberta-spam"
BATCH_SIZE = 16
PROJECT_ROOT_PATH = os.getcwd()
EMAIL_SPAM_CSV_PATH = os.path.join(PROJECT_ROOT_PATH, "email_spam.csv")

# Define the OBSERVED, CORRECT mapping for this specific model, overriding the config.
MANUAL_LABEL_MAP = {
    0: "spam",
    1: "ham"
}


def load_and_prep_data(source_name, source_type='hf', text_col='text', label_col='label', split='test', label_map=None):
    """
    A flexible function to load data from either Hugging Face or a local CSV.
    """
    print(f"\n- Loading and preparing data from: {source_name} ({split} split)")
    true_labels, eval_messages = [], []
    
    try:
        if source_type == 'hf':
            dataset = load_dataset(source_name, split=split)
            dataset = dataset.filter(lambda x: x[text_col] is not None and x[label_col] is not None and str(x[text_col]).strip() != "")
            true_labels_raw = dataset[label_col]
            eval_messages = dataset[text_col]
        
        elif source_type == 'local_csv':
            if not os.path.exists(source_name):
                print(f"❌ ERROR: Local file not found at '{source_name}'. Skipping.")
                return None, None
            df = pd.read_csv(source_name)
            df.dropna(subset=[text_col, label_col], inplace=True)
            true_labels_raw = df[label_col].astype(str).tolist()
            eval_messages = df[text_col].astype(str).tolist()

        # Standardize labels
        true_labels = [label.lower() for label in true_labels_raw]
        if label_map:
            true_labels = [label_map.get(label, label) for label in true_labels]

        print(f"✅ Loaded {len(eval_messages)} messages.")
        return true_labels, eval_messages

    except Exception as e:
        print(f"❌ ERROR: Failed to load or process data source {source_name}. Details: {e}")
        return None, None


def run_evaluation(model, tokenizer, device, true_labels, eval_messages, model_name, dataset_name):
    """
    Runs the core prediction loop and prints a standardized report.
    *** MODIFIED to use the manual label map. ***
    """
    print(f"- Making predictions on {dataset_name}...")
    all_predictions = []
    total_time_s = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(eval_messages), BATCH_SIZE), desc=f"Classifying {os.path.basename(dataset_name)}"):
            batch_messages = eval_messages[i:i + BATCH_SIZE]
            start_time = time.perf_counter()
            inputs = tokenizer(batch_messages, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            outputs = model(**inputs)
            predicted_indices = torch.argmax(outputs.logits, dim=1)
            total_time_s += (time.perf_counter() - start_time)
            
            # Use our empirically correct MANUAL_LABEL_MAP, not the flawed model.config.id2label
            batch_predictions = [MANUAL_LABEL_MAP[idx.item()] for idx in predicted_indices]
            all_predictions.extend(batch_predictions)

    print("✅ Predictions complete.")

    # Report the results
    print("\n" + "="*60)
    print(f"  DETAILED REPORT FOR: {model_name}")
    print(f"  ON TEST SET: {dataset_name}")
    print("="*60)

    accuracy = accuracy_score(true_labels, all_predictions)
    avg_time_ms = (total_time_s * 1000) / len(eval_messages)
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Total Prediction Time: {total_time_s:.4f} seconds")
    print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")
    
    all_known_labels = ['ham', 'spam']
    print("\nClassification Report:")
    print(classification_report(true_labels, all_predictions, labels=all_known_labels, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, all_predictions, labels=all_known_labels)
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='cividis', xticklabels=all_known_labels, yticklabels=all_known_labels)
    plt.title(f'{os.path.basename(model_name)}\n on {os.path.basename(dataset_name)}')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.show(block=False)


def evaluate_roberta_on_all():
    """
    Main function to load the RoBERTa model and evaluate it on all four specified datasets.
    """
    print("="*80)
    print(f"  STARTING FULL EVALUATION SUITE FOR: {MODEL_ID}")
    print("="*80)

    # --- 1. Load Model and Tokenizer (Once) ---
    print("\nLoading pre-trained RoBERTa model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device); model.eval()
        print(f"✅ Model loaded successfully on {device}.")
    except Exception as e:
        print(f"❌ ERROR: Could not load model from Hugging Face. Details: {e}"); return

    # --- 2. Define Evaluation Jobs ---
    evaluation_jobs = [
        {
            "dataset_name": "thehamkercat/telegram-spam-ham",
            "source_type": "hf", "split": "train",
            "text_col": "text", "label_col": "text_type",
            "label_map": None
        },
        {
            "dataset_name": "Deysi/spam-detection-dataset",
            "source_type": "hf", "split": "test",
            "text_col": "text", "label_col": "label",
            "label_map": {"not_spam": "ham"}
        },
        {
            "dataset_name": "SetFit/enron_spam",
            "source_type": "hf", "split": "test",
            "text_col": "message", "label_col": "label_text",
            "label_map": None
        },
        {
            "dataset_name": EMAIL_SPAM_CSV_PATH,
            "source_type": "local_csv", "split": None,
            "text_col": "text", "label_col": "type",
            "label_map": {"not spam": "ham"}
        }
    ]

    # --- 3. Run All Jobs ---
    for job in evaluation_jobs:
        true_labels, eval_messages = load_and_prep_data(
            source_name=job["dataset_name"],
            source_type=job["source_type"],
            text_col=job["text_col"],
            label_col=job["label_col"],
            split=job["split"],
            label_map=job["label_map"]
        )
        if true_labels and eval_messages:
            run_evaluation(model, tokenizer, device, true_labels, eval_messages, MODEL_ID, job["dataset_name"])

    print("\n" + "="*80)
    print("  ALL EVALUATIONS COMPLETE. SHOWING PLOTS.")
    print("="*80)
    plt.show()


if __name__ == "__main__":
    evaluate_roberta_on_all()