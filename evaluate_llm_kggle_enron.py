# evaluate_llm_email.py

import os
import sys
import time
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import load_dataset
import argparse

# --- Configuration ---
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# --- Path Configuration ---
PROJECT_ROOT_PATH = os.getcwd()
EMAIL_SPAM_CSV_PATH = os.path.join(PROJECT_ROOT_PATH, "email_spam.csv")


def construct_email_advanced_prompt(subject: str, message: str) -> list:
    """
    Builds a prompt specifically optimized for email classification.
    It combines the subject and body and provides email-specific few-shot examples.
    """
    system_prompt = (
        "You are an expert email spam detection classifier. Your task is to analyze the subject and body of the user's email. "
        "First, perform a step-by-step analysis to determine the email's intent (e.g., business correspondence, security notification, marketing, phishing attempt). "
        "Pay close attention to the tone, urgency, and any requested actions. After your analysis, on a new line, state your final classification as ONLY the single word 'spam' or 'ham'."
    )
    
    # Few-shot examples tailored for email formats
    few_shot_examples = [
        {
            "role": "user",
            "content": "Subject: Meeting tomorrow\n\nBody: Hi team, Just a reminder about our project sync meeting tomorrow at 10 AM. Please come prepared with your updates. Best, John"
        },
        {
            "role": "assistant",
            "content": "Analysis: The email has a professional subject line and a clear, concise body related to a scheduled work meeting. The tone is informational. This is a standard piece of business correspondence.\nham"
        },
        {
            "role": "user",
            "content": "Subject: Confidential Business Proposal\n\nBody: Dear Sir/Madam, We have a confidential investment opportunity that guarantees a 300% return in 90 days. We require a small upfront investment to secure your position. This is a limited-time offer. Please reply for more details."
        },
        {
            "role": "assistant",
            "content": "Analysis: The subject is vague and uses a lure ('Confidential'). The body promises an unrealistically high return on investment and pressures for an 'upfront investment' and a quick reply. This is a classic advance-fee financial scam.\nspam"
        }
    ]
    
    # Combine the new subject and message for the final prompt
    full_email_text = f"Subject: {subject}\n\nBody: {message}"
    final_user_message = {"role": "user", "content": full_email_text}
    
    return [{"role": "system", "content": system_prompt}] + few_shot_examples + [final_user_message]

def classify_with_lm_studio_email(client: OpenAI, subject: str, message: str, model_name: str) -> dict:
    """
    Classifies a single email using the model loaded in LM Studio.
    """
    messages = construct_email_advanced_prompt(subject, message)
    start_time = time.perf_counter()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=256 # Allow space for the reasoning step
        )
        
        full_response_text = response.choices[0].message.content.strip()
        lower_response = full_response_text.lower()
        
        # Robustly parse the final word from the CoT response
        last_ham_index = lower_response.rfind("ham")
        last_spam_index = lower_response.rfind("spam")

        if last_ham_index > last_spam_index:
            prediction_text = 'ham'
        elif last_spam_index > last_ham_index:
            prediction_text = 'spam'
        else:
            print(f"\nWarning: Could not parse 'ham' or 'spam' from response: '{full_response_text}'. Defaulting to 'ham'.")
            prediction_text = 'ham'

    except Exception as e:
        print(f"\nERROR: API call to LM Studio failed: {e}. Defaulting to 'ham'.")
        prediction_text = 'ham'

    end_time = time.perf_counter()
    
    return {"prediction": prediction_text, "time_ms": (end_time - start_time) * 1000}

def load_and_prep_data(source_name, source_type='hf'):
    """A flexible function to load and prepare the email datasets."""
    print(f"\n- Loading and preparing data from: {source_name}")
    subjects, messages, true_labels = [], [], []
    
    try:
        if source_type == 'hf':
            dataset = load_dataset(source_name, split='test')
            dataset = dataset.filter(lambda x: x['subject'] is not None and x['message'] is not None and x['label_text'] is not None)
            subjects = dataset['subject']; messages = dataset['message']
            true_labels = [label.lower() for label in dataset['label_text']]
        
        elif source_type == 'local_csv':
            if not os.path.exists(source_name):
                print(f"❌ ERROR: Local file not found at '{source_name}'."); return None, None, None
            df = pd.read_csv(source_name); df.dropna(subset=['title', 'text', 'type'], inplace=True)
            subjects = df['title'].astype(str).tolist(); messages = df['text'].astype(str).tolist()
            true_labels = df['type'].str.lower().replace('not spam', 'ham').tolist()

        print(f"✅ Loaded {len(messages)} messages.")
        return subjects, messages, true_labels

    except Exception as e:
        print(f"❌ ERROR: Failed to load/process {source_name}. Details: {e}"); return None, None, None


def evaluate_email_model(model_name: str):
    """Main function to run the evaluation suite."""
    print(f"\n--- Starting Email Evaluation for LM Studio Model: {model_name} ---")
    
    try:
        client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="not-needed"); client.models.list()
        print("✅ Successfully connected to LM Studio server.")
    except Exception:
        print(f"❌ FATAL ERROR: Could not connect to LM Studio. Please start the server."); return

    # Define the evaluation jobs
    evaluation_jobs = [
        {"source_name": "SetFit/enron_spam", "source_type": "hf"},
        {"source_name": EMAIL_SPAM_CSV_PATH, "source_type": "local_csv"}
    ]

    for job in evaluation_jobs:
        dataset_name = job["source_name"]
        subjects, messages, true_labels = load_and_prep_data(dataset_name, job["source_type"])
        
        if not true_labels:
            continue

        # Run Predictions
        results = []
        total_time = 0
        for subject, message in tqdm(zip(subjects, messages), total=len(messages), desc=f"Classifying {os.path.basename(dataset_name)}"):
            result = classify_with_lm_studio_email(client, subject, message, model_name)
            results.append(result)
            total_time += result['time_ms']
        
        predictions = [r['prediction'] for r in results]
        print("✅ Predictions complete.")

        # Report Results
        print("\n" + "="*60); print(f"  DETAILED REPORT FOR: {model_name} (Email Prompting)"); print(f"  ON TEST SET: {os.path.basename(dataset_name)}"); print("="*60)
        
        accuracy = accuracy_score(true_labels, predictions); avg_time_ms = total_time / len(results)
        
        print(f"\nOverall Accuracy: {accuracy:.2%}"); print(f"Total Prediction Time: {total_time / 1000:.2f} seconds"); print(f"Average Prediction Time: {avg_time_ms:.2f} ms/message")
        
        all_known_labels = ['ham', 'spam']
        print("\nClassification Report:"); print(classification_report(true_labels, predictions, labels=all_known_labels, zero_division=0))
        print("\nConfusion Matrix:"); cm = confusion_matrix(true_labels, predictions, labels=all_known_labels); print(cm)
        
        plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=all_known_labels, yticklabels=all_known_labels); plt.title(f'LM Studio - {model_name} (Email Prompt)\n on {os.path.basename(dataset_name)}'); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.show(block=False)

    print("\n" + "="*80); print("  ALL EMAIL EVALUATIONS COMPLETE. SHOWING PLOTS."); print("="*80); plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an email-classifying model running in LM Studio.")
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
            
    evaluate_email_model(model_to_test)