# ... (keep all imports from the original file)
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import joblib
import os

def retrain_and_save():
    """Function to perform the entire Naive Bayes training pipeline."""
    print("--- Starting Retraining Process ---")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    
    # --- 1. Load Data ---
    print("Loading data...")
    DATASET_PATH = "data/2cls_spam_text_cls.csv"
    df = pd.read_csv(
    "data/2cls_spam_text_cls.csv", 
    quotechar='"', 
    on_bad_lines='skip'
)
    # Drop rows with missing messages, which can happen with bad LLM data
    df.dropna(subset=['Message'], inplace=True)
    
    messages = df["Message"].astype(str).values.tolist()
    labels = df["Category"].values.tolist()

    # ... (The rest of the file is identical to the original train_nb.py)
    # --- 2. Preprocess Text ---
    print("Preprocessing text data...")
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def preprocess_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [stemmer.stem(token) for token in tokens]
        return tokens

    messages_processed = [preprocess_text(message) for message in messages]

    # --- 3. Feature Extraction (Bag-of-Words) ---
    print("Creating dictionary and feature vectors...")
    def create_dictionary(messages):
        all_words = set(word for message in messages for word in message)
        return sorted(list(all_words))

    dictionary = create_dictionary(messages_processed)

    def create_features(tokens, dictionary):
        features = np.zeros(len(dictionary))
        for token in tokens:
            if token in dictionary:
                features[dictionary.index(token)] += 1
        return features

    X = np.array([create_features(message, dictionary) for message in messages_processed])

    # --- 4. Process Labels ---
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # --- 5. Train Naive Bayes Model ---
    print("Training Naive Bayes model...")
    model = GaussianNB()
    model.fit(X, y)

    # --- 6. Save the Model and Associated Objects ---
    print("Saving model and artifacts...")
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, 'nb_model.joblib'))
    joblib.dump(dictionary, os.path.join(output_dir, 'dictionary.joblib'))
    joblib.dump(le, os.path.join(output_dir, 'label_encoder.joblib'))

    print("--- Retraining complete. ---")

# This allows the script to be run directly for initial training
if __name__ == "__main__":
    retrain_and_save()