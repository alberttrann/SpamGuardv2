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
DATA_DIR = os.path.join(BACKEND_DIR, 'data')


def retrain_and_save(dataset_filename: str):
    """
    Trains the superior MultinomialNB pipeline on the SPECIFIED dataset,
    creates a new versioned model, activates it, and updates the system
    configuration to use the dataset it was just trained on.
    """
    print(f"--- Starting Production Retraining on: {dataset_filename} ---")
    
    # --- 1. Load the Specified Dataset ---
    input_data_path = os.path.join(DATA_DIR, dataset_filename)
    if not os.path.exists(input_data_path):
        print(f"❌ ERROR: Dataset not found at '{input_data_path}'. Cannot retrain.")
        # We should raise an error here to stop the process in main.py
        raise FileNotFoundError(f"Dataset {dataset_filename} not found.")
        
    df = pd.read_csv(input_data_path, quotechar='"', on_bad_lines='skip')
    df.dropna(subset=['Message', 'Category'], inplace=True)
    df.drop_duplicates(subset=['Message'], inplace=True)
    
    X = df["Message"].astype(str)
    y_labels = df["Category"]
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    print(f"Data loaded successfully from '{dataset_filename}'.")

    # --- 2. Define Production Training Pipeline ---
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=preprocess_tokenizer,
            stop_words=None, # Handled in our custom tokenizer
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

    # --- 4. Save Artifacts and Update Registry ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"nb_multinomial_{timestamp}"
    
    pipeline_filename = f"{model_id}_pipeline.joblib"
    encoder_filename = f"{model_id}_encoder.joblib"
    
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(OUTPUT_MODELS_DIR, pipeline_filename))
    joblib.dump(le, os.path.join(OUTPUT_MODELS_DIR, encoder_filename))
    
    # Add the new model to the registry
    registry.add_model_to_registry(model_id, pipeline_filename, encoder_filename)
    # Set this newly trained model as the active one
    registry.set_active_model(model_id)
    
    current_config = registry.get_current_config()
    registry.set_current_config(current_config['mode'], dataset_filename)
    
    print(f"--- Retraining complete. ---")
    print(f"  - New model version '{model_id}' created and activated.")
    print(f"  - System config updated to use '{dataset_filename}' for k-NN indexing.")


if __name__ == "__main__":
    print("Running train_nb.py directly as a script.")
    default_dataset = "2cls_spam_text_cls.csv"
    print(f"Using default dataset for manual run: {default_dataset}")
    try:
        retrain_and_save(default_dataset)
    except FileNotFoundError as e:
        print(f"\nCould not run manual training: {e}")