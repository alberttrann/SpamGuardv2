# evaluate_lm_studio.py 

import os
import sys
import time
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse 

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
PROJECT_ROOT_PATH = os.getcwd()
EVALUATION_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "only_tricky_ham_test_set.txt")

def classify_with_lm_studio(client: OpenAI, message: str, model_name: str) -> dict:
    system_prompt = "You are an expert spam detection classifier. Analyze the user's message. Respond with ONLY the single word 'spam' or 'ham'. Do not add explanations or punctuation."
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": message}], temperature=0.0, max_tokens=5)
        prediction_text = response.choices[0].message.content.strip().lower()
        if prediction_text not in ['ham', 'spam']:
            print(f"\n--- LLM Failure Report ---"); print(f"  - Model: {model_name}"); print(f"  - Invalid Response: '{prediction_text}'"); print(f"  - Original Message: \"{message}\""); print(f"  - Action: Defaulting to 'ham'."); print(f"--------------------------")
            prediction_text = 'ham'
    except Exception as e:
        print(f"\n--- LLM Failure Report (API Error) ---"); print(f"  - Model: {model_name}"); print(f"  - Error: {e}"); print(f"  - Original Message: \"{message}\""); print(f"  - Action: Defaulting to 'ham'."); print(f"------------------------------------")
        prediction_text = 'ham'
    end_time = time.perf_counter()
    return {"prediction": prediction_text, "time_ms": (end_time - start_time) * 1000}

def evaluate_lm_studio_model(model_name: str):
    print(f"\n--- Starting Evaluation of LM Studio Model: {model_name} ---")
    try:
        client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="not-needed"); client.models.list()
        print("✅ Successfully connected to LM Studio server.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not connect to LM Studio server at {LM_STUDIO_BASE_URL}."); print("Please ensure you have loaded a model and clicked 'Start Server' in the LM Studio app.")
        return
    print(f"Loading evaluation data from '{EVALUATION_DATA_PATH}'...");
    with open(EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f: lines = f.readlines()
    true_labels = []; eval_messages = []
    for line in lines:
        if line.strip(): label, message = line.split(',', 1); true_labels.append(label.strip()); eval_messages.append(message.strip())
    print(f"✅ Loaded {len(eval_messages)} messages for evaluation.")
    print(f"\nMaking predictions using {model_name}. This may take a moment...")
    results = []
    total_time = 0
    for message in tqdm(eval_messages, desc=f"Classifying with {model_name}"):
        result = classify_with_lm_studio(client, message, model_name); results.append(result); total_time += result['time_ms']
    predictions = [r['prediction'] for r in results]
    print("✅ Predictions complete.")
    print("\n" + "="*60); print(f"  DETAILED REPORT FOR: {model_name} (via LM Studio)"); print("="*60)
    accuracy = accuracy_score(true_labels, predictions); avg_time_ms = total_time / len(results)
    print(f"\nOverall Accuracy: {accuracy:.2%}"); print(f"Total Prediction Time: {total_time / 1000:.4f} seconds"); print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")
    print("\nClassification Report:"); print(classification_report(true_labels, predictions, labels=['ham', 'spam'], zero_division=0))
    print("Confusion Matrix:"); cm = confusion_matrix(true_labels, predictions, labels=['ham', 'spam']); print(cm)
    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam']); plt.title(f'LM Studio - {model_name} Confusion Matrix'); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model running in LM Studio. You can either provide the model name as an argument or be prompted for it."
    )
    parser.add_argument(
        "model_name", 
        nargs='?', 
        default=None, 
        type=str, 
        help="Optional: The identifier for the model you are testing (e.g., 'Phi-3-mini-4k-instruct')."
    )
    args = parser.parse_args()

    model_to_test = args.model_name
    
    if not model_to_test:
        try:
            model_to_test = input("▶ Model name not provided. Please enter the model identifier currently loaded in LM Studio: ")
            if not model_to_test:
                print("❌ No model name entered. Exiting.")
                sys.exit(1) 
        except KeyboardInterrupt:
            print("\nOperation cancelled by user. Exiting.")
            sys.exit(0)

    evaluate_lm_studio_model(model_to_test)