# evaluate_llm_advanced.py

import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


EVALUATION_FILE_PATH = "only_tricky_ham_test_set.txt"
# EVALUATION_FILE_PATH = "only_tricky_ham_test_set.txt"

load_dotenv()
FPT_API_KEY = os.getenv("FPT_API_KEY")
FPT_BASE_URL = "https://mkp-api.fptcloud.com"
MODEL_NAME_FOR_REPORT = "DeepSeek-V3"
MODEL_ID_FOR_API = "DeepSeek-V3" 


def construct_advanced_prompt(message: str) -> list:
    """
    Builds a prompt that combines a detailed system instruction (CoT)
    with examples (Few-Shot) to guide the LLM's reasoning.
    """
    system_prompt = (
        "You are an expert spam detection classifier. Your task is to analyze the user's message. "
        "First, you will perform a step-by-step analysis to determine the message's intent. "
        "Consider if it is a transactional notification, a security alert, a marketing offer, or a phishing attempt. "
        "After your analysis, on a new line, state your final classification as ONLY the single word 'spam' or 'ham'."
    )
    
    few_shot_examples = [
        {"role": "user", "content": "Action required: Your account has been flagged for unusual login activity from a new device. Please verify your identity immediately."},
        {"role": "assistant", "content": ("Analysis: The message uses urgent keywords like 'Action required' and 'verify your identity immediately'. However, it describes a standard security procedure (flagging unusual login). This is a typical, legitimate security notification.\nham")},
        {"role": "user", "content": "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!"},
        {"role": "assistant", "content": ("Analysis: The message claims the user has won a high-value prize for no reason. It creates a sense of urgency ('claim now!') and requires a click. This is a classic promotional scam.\nspam")}
    ]
    
    final_user_message = {"role": "user", "content": message}
    
    return [{"role": "system", "content": system_prompt}] + few_shot_examples + [final_user_message]

def classify_with_advanced_prompt(client: OpenAI, message: str) -> dict:
    """
    Classifies a single message using the advanced prompt, handles parsing,
    and measures latency.
    """
    messages = construct_advanced_prompt(message)
    start_time = time.perf_counter()
    
    try:
        response = client.chat.completions.create(
            model=MODEL_ID_FOR_API,
            messages=messages,
            temperature=0.0,
            max_tokens=256 
        )
        
        full_response_text = response.choices[0].message.content.strip()
        lower_response = full_response_text.lower()
        
        last_ham_index = lower_response.rfind("ham")
        last_spam_index = lower_response.rfind("spam")

        if last_ham_index > last_spam_index:
            prediction_text = 'ham'
        elif last_spam_index > last_ham_index:
            prediction_text = 'spam'
        else:
            print(f"\n--- LLM Failure Report (Parsing) ---")
            print(f"  - Model: {MODEL_NAME_FOR_REPORT}")
            print(f"  - Could not find definitive 'ham' or 'spam' in response: '{full_response_text}'")
            print(f"  - Original Message: \"{message}\"")
            print(f"  - Action: Defaulting to 'ham' (safe choice).")
            print(f"------------------------------------")
            prediction_text = 'ham'

    except Exception as e:
        print(f"\n--- LLM Failure Report (API Error) ---")
        print(f"  - Model: {MODEL_NAME_FOR_REPORT}")
        print(f"  - Error: {e}")
        print(f"  - Original Message: \"{message}\"")
        print(f"  - Action: Defaulting to 'ham'.")
        print(f"------------------------------------")
        prediction_text = 'ham'
        
    end_time = time.perf_counter()
    return {"prediction": prediction_text, "time_ms": (end_time - start_time) * 1000}

def evaluate_llm_advanced():
    """
    Main function to run the evaluation using the configured model and test set.
    """
    test_set_name = os.path.basename(EVALUATION_FILE_PATH)
    print(f"--- Starting ADVANCED PROMPT Evaluation of {MODEL_NAME_FOR_REPORT} on {test_set_name} ---")

    if not FPT_API_KEY:
        print("❌ FATAL ERROR: FPT_API_KEY environment variable not set."); return
        
    client = OpenAI(api_key=FPT_API_KEY, base_url=FPT_BASE_URL)

    if not os.path.exists(EVALUATION_FILE_PATH):
        print(f"❌ ERROR: Evaluation file not found at '{EVALUATION_FILE_PATH}'."); return
        
    with open(EVALUATION_FILE_PATH, 'r', encoding='utf-8') as f: lines = f.readlines()
    
    true_labels = []; eval_messages = []
    for line in lines:
        if line.strip():
            parts = line.split(',', 1)
            if len(parts) == 2 and parts[0].strip().lower() in ['ham', 'spam']:
                true_labels.append(parts[0].strip().lower()); eval_messages.append(parts[1].strip())

    print(f"✅ Loaded {len(eval_messages)} messages for evaluation.")

    print(f"\nMaking predictions using {MODEL_NAME_FOR_REPORT}. This will take several minutes...")
    results = []; total_time = 0
    for message in tqdm(eval_messages, desc=f"Classifying with {MODEL_NAME_FOR_REPORT}"):
        result = classify_with_advanced_prompt(client, message)
        results.append(result)
        total_time += result['time_ms']
        time.sleep(0.2) 

    predictions = [r['prediction'] for r in results]
    print("✅ Predictions complete.")

    # Report Results
    print("\n" + "="*60)
    print(f"  DETAILED REPORT FOR: {MODEL_NAME_FOR_REPORT} (Advanced Prompting)")
    print(f"  ON TEST SET: {test_set_name}")
    print("="*60)
    
    accuracy = accuracy_score(true_labels, predictions)
    avg_time_ms = total_time / len(results)
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Total Prediction Time: {total_time / 1000:.2f} seconds")
    print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")
    
    # Ensure a 2x2 report even if one class is missing (like in the tricky ham set)
    all_known_labels = ['ham', 'spam']
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, labels=all_known_labels, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions, labels=all_known_labels)
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=all_known_labels, yticklabels=all_known_labels)
    plt.title(f'{MODEL_NAME_FOR_REPORT} (Advanced Prompt)\n on {test_set_name}')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.show()

if __name__ == "__main__":
    evaluate_llm_advanced()