
import os
import sys
import time
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import load_dataset, concatenate_datasets
import argparse

# --- Configuration ---
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

def construct_shortform_advanced_prompt(message: str) -> list:
    """
    Builds a prompt that combines Chain-of-Thought with Few-Shot examples
    tailored for SMS, Telegram, or general web text.
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

def classify_with_lm_studio(client: OpenAI, message: str, model_name: str) -> dict:
    """Classifies a single message using the model loaded in LM Studio."""
    messages = construct_shortform_advanced_prompt(message)
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=1024
        )
        full_response_text = response.choices[0].message.content.strip()
        lower_response = full_response_text.lower()
        last_ham_index = lower_response.rfind("ham"); last_spam_index = lower_response.rfind("spam")

        if last_ham_index > last_spam_index: prediction_text = 'ham'
        elif last_spam_index > last_ham_index: prediction_text = 'spam'
        else:
            print(f"\nWarning: Could not parse 'ham' or 'spam' from response: '{full_response_text}'. Defaulting to 'ham'.")
            prediction_text = 'ham'
    except Exception as e:
        print(f"\nERROR: API call to LM Studio failed: {e}. Defaulting to 'ham'.")
        prediction_text = 'ham'
    end_time = time.perf_counter()
    return {"prediction": prediction_text, "time_ms": (end_time - start_time) * 1000}

def load_and_prep_data(job_config):
    """A flexible function to load and prepare the datasets based on a config dict."""
    source_name = job_config["source_name"]
    print(f"\n- Loading and preparing data from: {source_name}")
    
    try:
        dataset = load_dataset(source_name, split=job_config["split"])
        dataset = dataset.filter(lambda x: x[job_config["text_col"]] is not None and x[job_config["label_col"]] is not None and str(x[job_config["text_col"]]).strip() != "")

        if source_name == "thehamkercat/telegram-spam-ham":
            print("Creating a balanced 2000-sample subset for 'telegram-spam-ham'...")
            ham_data = dataset.filter(lambda x: x[job_config["label_col"]] == "ham").shuffle(seed=42).select(range(1000))
            spam_data = dataset.filter(lambda x: x[job_config["label_col"]] == "spam").shuffle(seed=42).select(range(1000))
            dataset = concatenate_datasets([ham_data, spam_data]).shuffle(seed=42)
            print("✅ Subset created successfully.")

        true_labels_raw = dataset[job_config["label_col"]]
        eval_messages = dataset[job_config["text_col"]]
        
        label_map = job_config.get("label_map")
        true_labels = [label.lower() for label in true_labels_raw]
        if label_map:
            true_labels = [label_map.get(label, label) for label in true_labels]

        print(f"✅ Loaded {len(eval_messages)} messages for evaluation.")
        return eval_messages, true_labels
    except Exception as e:
        print(f"❌ ERROR: Failed to load/process {source_name}. Details: {e}"); return None, None


def evaluate_shortform_model(model_name: str):
    """Main function to run the evaluation suite."""
    print(f"\n--- Starting Short-Form Text Evaluation for LM Studio Model: {model_name} ---")
    
    try:
        client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="not-needed"); client.models.list()
        print("✅ Successfully connected to LM Studio server.")
    except Exception:
        print(f"❌ FATAL ERROR: Could not connect to LM Studio. Please start the server."); return

    # Define the evaluation jobs
    evaluation_jobs = [
        {"source_name": "thehamkercat/telegram-spam-ham", "split": "train", "text_col": "text", "label_col": "text_type"},
        {"source_name": "Deysi/spam-detection-dataset", "split": "test", "text_col": "text", "label_col": "label", "label_map": {"not_spam": "ham"}}
    ]

    for job in evaluation_jobs:
        dataset_name = job["source_name"]
        eval_messages, true_labels = load_and_prep_data(job)
        
        if not true_labels:
            continue

        # Run Predictions
        results = []
        total_time = 0
        for message in tqdm(eval_messages, desc=f"Classifying {os.path.basename(dataset_name)}"):
            result = classify_with_lm_studio(client, message, model_name)
            results.append(result)
            total_time += result['time_ms']
        
        predictions = [r['prediction'] for r in results]
        print("✅ Predictions complete.")

        # Report Results
        print("\n" + "="*60); print(f"  DETAILED REPORT FOR: {model_name} (Advanced Prompt)"); print(f"  ON TEST SET: {os.path.basename(dataset_name)}"); print("="*60)
        
        accuracy = accuracy_score(true_labels, predictions); avg_time_ms = total_time / len(results)
        
        print(f"\nOverall Accuracy: {accuracy:.2%}"); print(f"Total Prediction Time: {total_time / 1000:.2f} seconds"); print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")
        
        all_known_labels = ['ham', 'spam']
        print("\nClassification Report:"); print(classification_report(true_labels, predictions, labels=all_known_labels, zero_division=0))
        print("\nConfusion Matrix:"); cm = confusion_matrix(true_labels, predictions, labels=all_known_labels); print(cm)
        
        plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=all_known_labels, yticklabels=all_known_labels); plt.title(f'LM Studio - {model_name} (Advanced Prompt)\n on {os.path.basename(dataset_name)}'); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.show(block=False)

    print("\n" + "="*80); print("  ALL SHORT-FORM EVALUATIONS COMPLETE. SHOWING PLOTS."); print("="*80); plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a short-form text classifying model running in LM Studio.")
    parser.add_argument("model_name", nargs='?', default=None, type=str, help="Optional: Identifier for the model to test.")
    args = parser.parse_args()
    
    model_to_test = args.model_name
    if not model_to_test:
        try:
            model_to_test = input("▶ Model name not provided. Please enter the model identifier loaded in LM Studio: ")
            if not model_to_test:
                print("❌ No model name entered. Exiting."); sys.exit(1)
        except KeyboardInterrupt:
            print("\nOperation cancelled. Exiting."); sys.exit(0)
            
    evaluate_shortform_model(model_to_test)