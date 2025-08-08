# evaluate_lm_studio_simple_prompt.py

import os
import sys
import time
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse 

# --- Configuration ---
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
PROJECT_ROOT_PATH = os.getcwd()
# --- EDIT THIS LINE to change the test set ---
EVALUATION_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "only_tricky_ham_test_set.txt")
# EVALUATION_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "mixed_test_set.txt")

def classify_with_lm_studio_simple(client: OpenAI, message: str, model_name: str) -> dict:
    """
    Classifies a message using a simpler prompt but with robust parsing.
    """
    # This prompt is simpler. It doesn't ask for a step-by-step analysis,
    # but it still strongly guides the model to the desired output format.
    system_prompt = (
        "You are an expert spam detection classifier. Your task is to analyze the user's message. "
        "Conclude your response with a final verdict on a new line, containing ONLY the single word 'spam' or 'ham'."
    )
    
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.0,
            max_tokens=150 # Increased slightly to allow for some verbosity
        )
        
        # --- Robust Parsing Logic ---
        full_response_text = response.choices[0].message.content.strip()
        lower_response = full_response_text.lower()
        
        # Find the last occurrence of 'ham' or 'spam'
        last_ham_index = lower_response.rfind("ham")
        last_spam_index = lower_response.rfind("spam")

        # The last one found is the final verdict
        if last_ham_index > last_spam_index:
            prediction_text = 'ham'
        elif last_spam_index > last_ham_index:
            prediction_text = 'spam'
        else:
            # If neither word is found, the model failed to follow instructions
            print(f"\n--- LLM Failure Report (Parsing) ---")
            print(f"  - Model: {model_name}")
            print(f"  - Could not find a definitive 'ham' or 'spam' in response: '{full_response_text}'")
            print(f"  - Original Message: \"{message}\"")
            print(f"  - Action: Defaulting to 'ham'.")
            print(f"------------------------------------")
            prediction_text = 'ham'

    except Exception as e:
        print(f"\n--- LLM Failure Report (API Error) ---"); print(f"  - Model: {model_name}"); print(f"  - Error: {e}"); print(f"  - Original Message: \"{message}\""); print(f"  - Action: Defaulting to 'ham'."); print(f"------------------------------------")
        prediction_text = 'ham'
        
    end_time = time.perf_counter()
    return {"prediction": prediction_text, "time_ms": (end_time - start_time) * 1000}

def evaluate_lm_studio_model(model_name: str):
    test_set_name = os.path.basename(EVALUATION_DATA_PATH)
    print(f"\n--- Starting SIMPLE PROMPT Evaluation of {model_name} on {test_set_name} ---")
    
    try:
        client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="not-needed"); client.models.list()
        print("✅ Successfully connected to LM Studio server.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not connect to LM Studio server. Is it running?"); return

    print(f"Loading evaluation data from '{EVALUATION_DATA_PATH}'...");
    with open(EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f: lines = f.readlines()
    
    true_labels = []; eval_messages = []
    for line in lines:
        if line.strip(): label, message = line.split(',', 1); true_labels.append(label.strip().lower()); eval_messages.append(message.strip())
        
    print(f"✅ Loaded {len(eval_messages)} messages for evaluation.")
    
    print(f"\nMaking predictions using {model_name}...")
    results = []
    total_time = 0
    for message in tqdm(eval_messages, desc=f"Classifying with {model_name}"):
        result = classify_with_lm_studio_simple(client, message, model_name); results.append(result); total_time += result['time_ms']
        
    predictions = [r['prediction'] for r in results]
    print("✅ Predictions complete.")
    
    print("\n" + "="*60); print(f"  DETAILED REPORT FOR: {model_name} (Simple Prompt)"); print(f"  ON TEST SET: {test_set_name}"); print("="*60)
    
    accuracy = accuracy_score(true_labels, predictions); avg_time_ms = total_time / len(results)
    print(f"\nOverall Accuracy: {accuracy:.2%}"); print(f"Total Prediction Time: {total_time / 1000:.4f} seconds"); print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")
    
    unique_labels = sorted(list(set(true_labels)))
    print("\nClassification Report:"); print(classification_report(true_labels, predictions, labels=['ham', 'spam'], zero_division=0))
    
    print("\nConfusion Matrix:"); cm = confusion_matrix(true_labels, predictions, labels=['ham', 'spam']); print(cm)
    
    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam']); plt.title(f'LM Studio - {model_name} (Simple Prompt)\n on {test_set_name}'); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model in LM Studio with a simple prompt.")
    parser.add_argument("model_name", nargs='?', default=None, type=str, help="Optional: The model identifier.")
    args = parser.parse_args()
    model_to_test = args.model_name
    
    if not model_to_test:
        try:
            model_to_test = input("▶ Model name not provided. Please enter the model identifier currently loaded in LM Studio: ")
            if not model_to_test: print("❌ No model name entered. Exiting."); sys.exit(1) 
        except KeyboardInterrupt: print("\nOperation cancelled. Exiting."); sys.exit(0)

    evaluate_lm_studio_model(model_to_test)