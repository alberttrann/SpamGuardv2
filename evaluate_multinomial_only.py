# evaluate_multinomial_only.py 

import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT_PATH = os.getcwd()

if PROJECT_ROOT_PATH not in sys.path:
    sys.path.append(PROJECT_ROOT_PATH)

from backend.utils import preprocess_tokenizer

ORIGINAL_DATA_PATH = r"C:\Users\alberttran\Downloads\2cls_spam_text_cls_original.csv"
CURRENT_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'backend', 'data', '2cls_spam_text_cls.csv')
EVALUATION_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "evaluation_data.txt")


def train_and_evaluate_multinomial(dataset_path: str, dataset_name: str):
    """
    Trains a full MultinomialNB pipeline on the given dataset and then
    evaluates it against the standard evaluation file.
    """
    print("-" * 60)
    print(f"--- Evaluating MultinomialNB ONLY Model on: {dataset_name} ---")

    # --- 1. Load Training Data ---
    print(f"Loading training data from: {os.path.basename(dataset_path)}")
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset file not found at '{dataset_path}'.")
        return
        
    df = pd.read_csv(dataset_path, quotechar='"', on_bad_lines='skip')
    df.dropna(subset=['Message', 'Category'], inplace=True)
    df.drop_duplicates(subset=['Message'], inplace=True)
    
    X_train = df["Message"].astype(str)
    y_train_labels = df["Category"]
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_labels)
    
    # --- 2. Define and Train the Pipeline ---
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=preprocess_tokenizer, stop_words=None, ngram_range=(1, 2), max_features=10000)),
        ('smote', SMOTE(random_state=42)),
        ('clf', MultinomialNB(alpha=0.1))
    ])
    
    print("Training the MultinomialNB pipeline...")
    pipeline.fit(X_train, y_train)
    print("✅ Training complete.")

    # --- 3. Load Evaluation Data ---
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
            
    # --- 4. Make Predictions ---
    print("Making predictions on the evaluation set...")
    numeric_predictions = pipeline.predict(eval_messages)
    string_predictions = le.inverse_transform(numeric_predictions)
    print("✅ Predictions complete.")

    # --- 5. Report the results ---
    print(f"\n--- Evaluation Report for: {dataset_name} ---")
    accuracy = accuracy_score(true_labels, string_predictions)
    print(f"\nOverall Accuracy: {accuracy:.2%}\n")
    
    print("Classification Report:")
    report = classification_report(true_labels, string_predictions, target_names=le.classes_, zero_division=0)
    print(report)
    
    print("Confusion Matrix:")
    cm = confusion_matrix(true_labels, string_predictions, labels=le.classes_)
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'MultinomialNB Only Confusion Matrix\n(Trained on {dataset_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show(block=False)


if __name__ == "__main__":
    train_and_evaluate_multinomial(ORIGINAL_DATA_PATH, "Original Biased Dataset")
    train_and_evaluate_multinomial(CURRENT_DATA_PATH, "Current Augmented Dataset")
    
    print("\n" + "="*60)
    print("All evaluations complete. Showing plots.")
    plt.show()