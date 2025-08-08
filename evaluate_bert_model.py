# evaluate_bert_model.py (Corrected Version with Label Mapping)

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

EVALUATION_FILE_PATH = "evaluation_data.txt"
BATCH_SIZE = 16
MODEL_ID = "AventIQ-AI/bert-spam-detection"

LABEL_MAP = {
    "LABEL_0": "ham",
    "LABEL_1": "spam"
}


def load_test_data(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Evaluation file not found at '{file_path}'."); return None, None
    with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    true_labels = []; eval_messages = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        try:
            label, message = line.split(',', 1)
            label = label.strip().lower()
            if label not in ['ham', 'spam']: continue
            true_labels.append(label); eval_messages.append(message.strip())
        except ValueError: continue
    return true_labels, eval_messages


def evaluate_bert_classifier():
    test_set_name = os.path.basename(EVALUATION_FILE_PATH)
    print("="*60); print(f"  STARTING EVALUATION OF: {MODEL_ID}"); print(f"  ON TEST SET: {test_set_name}"); print("="*60)

    # --- 1. Load Model and Tokenizer ---
    print("\nLoading pre-trained BERT model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device); model.eval()
        print(f"✅ Model loaded successfully on {device}.")
    except Exception as e:
        print(f"❌ ERROR: Could not load model. Details: {e}"); return

    # --- 2. Load Evaluation Data ---
    print(f"\nLoading evaluation data from '{EVALUATION_FILE_PATH}'...")
    true_labels, eval_messages = load_test_data(EVALUATION_FILE_PATH)
    if not true_labels: print("❌ No valid data to evaluate. Exiting."); return
    print(f"✅ Loaded {len(eval_messages)} messages for evaluation.")

    # --- 3. Make Predictions in Batches ---
    print("\nMaking predictions...")
    all_predictions = []
    total_time_s = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(eval_messages), BATCH_SIZE), desc="Classifying Batches"):
            batch_messages = eval_messages[i:i + BATCH_SIZE]
            start_time = time.perf_counter()
            inputs = tokenizer(batch_messages, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            outputs = model(**inputs)
            predicted_indices = torch.argmax(outputs.logits, dim=1)
            total_time_s += (time.perf_counter() - start_time)
            
            # Step A: Get the model's native labels (e.g., 'LABEL_0')
            native_predictions = [model.config.id2label[idx.item()] for idx in predicted_indices]
            # Step B: Translate them to our standard labels ('ham', 'spam') using the map
            translated_predictions = [LABEL_MAP.get(pred, "ham") for pred in native_predictions] # Default to 'ham' if a weird label appears
            
            all_predictions.extend(translated_predictions)

    print("✅ Predictions complete.")

    # --- 4. Report the Results  ---
    print("\n" + "="*60); print(f"  DETAILED REPORT FOR: {MODEL_ID}"); print(f"  ON TEST SET: {test_set_name}"); print("="*60)
    accuracy = accuracy_score(true_labels, all_predictions)
    avg_time_ms = (total_time_s * 1000) / len(eval_messages)
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Total Prediction Time: {total_time_s:.4f} seconds")
    print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")
    
    all_known_labels = ['ham', 'spam']
    print("\nClassification Report:"); print(classification_report(true_labels, all_predictions, labels=all_known_labels, zero_division=0))
    print("\nConfusion Matrix:"); cm = confusion_matrix(true_labels, all_predictions, labels=all_known_labels); print(cm)
    
    # --- 5. Visualize the results ---
    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='cividis', xticklabels=all_known_labels, yticklabels=all_known_labels)
    plt.title(f'BERT ({MODEL_ID})\n on {test_set_name}'); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.show()

if __name__ == "__main__":
    evaluate_bert_classifier()