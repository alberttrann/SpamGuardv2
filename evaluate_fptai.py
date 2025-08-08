# evaluate_llm.py

import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# 1. API Configuration 
load_dotenv()
FPT_API_KEY = os.getenv("FPT_API_KEY")
FPT_BASE_URL = "https://mkp-api.fptcloud.com"
MODEL_ID = "Qwen2.5-7B-Instruct" 

# 2. Path Configuration
PROJECT_ROOT_PATH = os.getcwd()
EVALUATION_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "mixed_test_set.txt")

# --- LLM Classification Logic ---

def classify_with_llm(client: OpenAI, message: str) -> dict:
    """
    Classifies a single message using the LLM, handles parsing, and measures latency.
    """
    system_prompt = "You are an expert spam detection classifier. Your task is to analyze the user's message. Respond with ONLY the single word 'spam' or 'ham' and nothing else. Do not add explanations or punctuation."
    
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.0,  
            max_tokens=5      
        )
        
        # Extract, clean, and validate the model's prediction
        prediction_text = response.choices[0].message.content.strip().lower()
        if prediction_text not in ['ham', 'spam']:
            print(f"\nWarning: LLM returned an invalid response: '{prediction_text}'. Defaulting to 'ham'.")
            prediction_text = 'ham' # Default to the safer option in case of failure

    except Exception as e:
        print(f"\nERROR: An API call failed: {e}. Defaulting to 'ham'.")
        prediction_text = 'ham'

    end_time = time.perf_counter()
    
    return {
        "prediction": prediction_text,
        "time_ms": (end_time - start_time) * 1000
    }

def evaluate_llm_performance():
    """
    Main function to run the full evaluation suite against the LLM.
    """
    print("--- Starting Evaluation of LLM (DeepSeek-V2) ---")

    if not FPT_API_KEY:
        print("❌ FATAL ERROR: FPT_API_KEY environment variable not set.")
        print("Please set it before running the script (e.g., $env:FPT_API_KEY='your-key')")
        return
        
    # Initialize the API client
    client = OpenAI(api_key=FPT_API_KEY, base_url=FPT_BASE_URL)

    # --- 1. Load Evaluation Data ---
    print(f"Loading evaluation data from '{EVALUATION_DATA_PATH}'...")
    if not os.path.exists(EVALUATION_DATA_PATH):
        print(f"❌ ERROR: Evaluation data file not found at '{EVALUATION_DATA_PATH}'.")
        return
        
    with open(EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f: lines = f.readlines()
    
    true_labels = []
    eval_messages = []
    for line in lines:
        if line.strip():
            label, message = line.split(',', 1)
            true_labels.append(label.strip())
            eval_messages.append(message.strip())
    
    print(f"✅ Loaded {len(eval_messages)} messages for evaluation.")

    # --- 2. Make Predictions ---
    print("\nMaking predictions using the LLM. This may take several minutes...")
    results = []
    total_time = 0
    # Use tqdm for a nice progress bar
    for message in tqdm(eval_messages, desc="Classifying with LLM"):
        result = classify_with_llm(client, message)
        results.append(result)
        total_time += result['time_ms']
        time.sleep(0.5) 

    predictions = [r['prediction'] for r in results]
    print("✅ Predictions complete.")

    # --- 3. Report the results ---
    print("\n" + "="*60)
    print(f"  DETAILED REPORT FOR: LLM ({MODEL_ID})")
    print("="*60)
    
    accuracy = accuracy_score(true_labels, predictions)
    avg_time_ms = total_time / len(results)
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Total Prediction Time: {total_time / 1000:.4f} seconds")
    print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")
    
    print("\nClassification Report:")
    report = classification_report(true_labels, predictions, labels=['ham', 'spam'], zero_division=0)
    print(report)
    
    print("Confusion Matrix:")
    cm = confusion_matrix(true_labels, predictions, labels=['ham', 'spam'])
    print(cm)
    
    # --- 4. Visualize the results ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title(f'LLM ({MODEL_ID}) Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    evaluate_llm_performance()