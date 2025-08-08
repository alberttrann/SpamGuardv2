# evaluate_llm_advanced_prompts.py

import os
import sys
import time
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re

# --- Configuration ---
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
PROJECT_ROOT_PATH = os.getcwd()

# --- NEW: Define the paths to your test sets ---
# We will test on both to see the impact.
MIXED_TEST_SET_PATH = os.path.join(PROJECT_ROOT_PATH, "mixed_test_set.txt")
TRICKY_HAM_TEST_SET_PATH = os.path.join(PROJECT_ROOT_PATH, "only_tricky_ham_test_set.txt")

# --- Advanced Prompting Logic ---

def construct_advanced_prompt(message: str) -> list:
    """
    Builds a prompt that incorporates both Few-Shot examples and Chain-of-Thought instructions.
    """
    
    # System prompt sets the overall behavior and goal.
    system_prompt = (
        "You are an expert spam detection classifier. Your task is to analyze the user's message. "
        "First, you will perform a step-by-step analysis to determine the message's intent. "
        "Consider if it is a transactional notification, a security alert, a marketing offer, or a phishing attempt. "
        "After your analysis, on a new line, state your final classification as ONLY the single word 'spam' or 'ham'."
    )
    
    # Few-shot examples teach the model how to handle ambiguous cases.
    few_shot_examples = [
        {
            "role": "user",
            "content": "Action required: Your account has been flagged for unusual login activity from a new device. Please verify your identity immediately."
        },
        {
            "role": "assistant",
            "content": (
                "Analysis: The message uses urgent keywords like 'Action required' and 'verify your identity immediately'. "
                "However, it describes a standard security procedure (flagging unusual login). This is a typical, legitimate security notification.\n"
                "ham"
            )
        },
        {
            "role": "user",
            "content": "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!"
        },
        {
            "role": "assistant",
            "content": (
                "Analysis: The message claims the user has won a high-value prize for no reason. It creates a sense of urgency ('claim now!') and requires a click. "
                "This is a classic promotional scam.\n"
                "spam"
            )
        }
    ]
    
    # The final user message to be classified.
    final_user_message = {"role": "user", "content": message}
    
    # Combine everything into the final message list for the API
    return [{"role": "system", "content": system_prompt}] + few_shot_examples + [final_user_message]

def classify_with_advanced_prompt(client: OpenAI, message: str, model_name: str) -> dict:
    """
    Classifies a message using the advanced CoT + Few-Shot prompt.
    *** MODIFIED with more robust parsing logic. ***
    """
    messages = construct_advanced_prompt(message)
    start_time = time.perf_counter()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=200
        )
        
        full_response_text = response.choices[0].message.content.strip()
        
        # --- THIS IS THE ROBUST PARSING FIX ---
        # 1. Convert the entire response to lower case for consistency.
        lower_response = full_response_text.lower()
        
        # 2. Find the last index of "ham" and "spam".
        last_ham_index = lower_response.rfind("ham")
        last_spam_index = lower_response.rfind("spam")

        # 3. Determine which one appeared last.
        if last_ham_index > last_spam_index:
            prediction_text = 'ham'
        elif last_spam_index > last_ham_index:
            prediction_text = 'spam'
        else:
            # This case happens if neither word is found, or if one is found at index -1.
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
    
    return {
        "prediction": prediction_text,
        "time_ms": (end_time - start_time) * 1000
    }

def evaluate_llm_with_advanced_prompts(model_name: str, test_set_path: str):
    """
    Main evaluation function.
    *** MODIFIED with cleaned-up reporting logic. ***
    """
    test_set_name = os.path.basename(test_set_path)
    print(f"\n--- Starting ADVANCED PROMPT Evaluation of {model_name} on {test_set_name} ---")

    try:
        client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="not-needed"); client.models.list()
        print("✅ Successfully connected to LM Studio server.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not connect to LM Studio server at {LM_STUDIO_BASE_URL}."); print("Please ensure you have loaded a model and clicked 'Start Server'.")
        return

    # --- Load Evaluation Data ---
    print(f"Loading evaluation data from '{test_set_path}'...")
    with open(test_set_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    
    true_labels = []; eval_messages = []
    for line in lines:
        if line.strip():
            parts = line.split(',', 1)
            if len(parts) == 2:
                label, message = parts
                if label.strip().lower() in ['ham', 'spam']:
                    true_labels.append(label.strip().lower()); eval_messages.append(message.strip())

    if not eval_messages:
        print("❌ No valid messages found in the evaluation file.")
        return
        
    print(f"✅ Loaded {len(eval_messages)} messages for evaluation.")

    # --- Make Predictions ---
    results = []; total_time = 0
    for message in tqdm(eval_messages, desc=f"Classifying with {model_name}"):
        result = classify_with_advanced_prompt(client, message, model_name); results.append(result); total_time += result['time_ms']
    
    predictions = [r['prediction'] for r in results]
    print("✅ Predictions complete.")

    # --- THIS IS THE CLEANED-UP REPORTING BLOCK ---
    print("\n" + "="*60); print(f"  DETAILED REPORT FOR: {model_name} (Advanced Prompting)"); print(f"  ON TEST SET: {test_set_name}"); print("="*60)
    
    # Calculate all metrics once
    accuracy = accuracy_score(true_labels, predictions)
    avg_time_ms = total_time / len(results)
    unique_labels = sorted(list(set(true_labels)))
    report = classification_report(true_labels, predictions, labels=unique_labels, zero_division=0)
    cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
    
    # Print each metric once
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Total Prediction Time: {total_time / 1000:.4f} seconds")
    print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")
    
    print("\nClassification Report:")
    print(report)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualize the results once
    plt.figure(figsize=(8, 6)); 
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=unique_labels, yticklabels=unique_labels); 
    plt.title(f'LM Studio - {model_name} (Advanced Prompt)\n on {test_set_name}'); 
    plt.ylabel('True Label'); 
    plt.xlabel('Predicted Label'); 
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model in LM Studio with advanced prompting.")
    parser.add_argument("model_name", type=str, help="The identifier for the model you are testing.")
    args = parser.parse_args()
    
    # Run the evaluation on both test sets
    evaluate_llm_with_advanced_prompts(args.model_name, MIXED_TEST_SET_PATH)
    evaluate_llm_with_advanced_prompts(args.model_name, TRICKY_HAM_TEST_SET_PATH)