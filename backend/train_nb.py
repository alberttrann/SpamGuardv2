# backend/train_nb.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
from datetime import datetime

from .utils import preprocess_tokenizer
from . import registry

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, '..'))
OUTPUT_MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
INPUT_DATA_PATH = os.path.join(BACKEND_DIR, 'data', '2cls_spam_text_cls.csv')


def retrain_and_save():
    """
    This is the main retraining function for the SpamGuard application.
    It trains the superior MultinomialNB pipeline on the application's current dataset
    and saves the production-ready model artifacts with versioning.
    """
    print("--- Starting Production Retraining Process with Model Versioning ---")
    
    # --- 1. Load the Application's Current Dataset ---
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"❌ ERROR: Main dataset not found at '{INPUT_DATA_PATH}'. Cannot retrain.")
        return

    df = pd.read_csv(INPUT_DATA_PATH, quotechar='"', on_bad_lines='skip')
    df.dropna(subset=['Message', 'Category'], inplace=True)
    df.drop_duplicates(subset=['Message'], inplace=True)
    
    X = df["Message"].astype(str)
    y_labels = df["Category"]
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    print(f"Data loaded. Current distribution: {pd.Series(y_labels).value_counts().to_dict()}")

    # --- 2. Define the Production Training Pipeline ---
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=preprocess_tokenizer,
            stop_words=None, 
            ngram_range=(1, 2),
            max_features=10000,
        )),
        ('smote', SMOTE(random_state=42)),
        ('clf', MultinomialNB(alpha=0.1))
    ])

    # --- 3. Train the Model ---
    print("Training the production MultinomialNB pipeline...")
    pipeline.fit(X, y)
    print("✅ Production model training complete.")

    # --- 4. Save with a timestamp and update the registry ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"nb_multinomial_{timestamp}"
    
    pipeline_filename = f"{model_id}_pipeline.joblib"
    encoder_filename = f"{model_id}_encoder.joblib"
    
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(OUTPUT_MODELS_DIR, pipeline_filename))
    joblib.dump(le, os.path.join(OUTPUT_MODELS_DIR, encoder_filename))
    
    registry.add_model_to_registry(model_id, pipeline_filename, encoder_filename)
    registry.set_active_model(model_id)
    
    print(f"--- Retraining complete. New model version '{model_id}' created and activated. ---")

if __name__ == "__main__":
    retrain_and_save()