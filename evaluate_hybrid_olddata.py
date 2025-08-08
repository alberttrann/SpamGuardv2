# train_on_biased_data.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string


ORIGINAL_DATA_PATH = r"C:\Users\alberttran\Downloads\2cls_spam_text_cls_original.csv"

OUTPUT_MODEL_DIR = "models_trained_on_biased_data"

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()
english_stop_words = set(stopwords.words('english'))

def preprocess_tokenizer(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    filtered_and_stemmed_tokens = [
        stemmer.stem(token) for token in tokens if token.isalpha() and token not in english_stop_words
    ]
    return filtered_and_stemmed_tokens

def train_superior_model_on_biased_data():
    """
    Trains the superior V2 architecture (MultinomialNB + TF-IDF + SMOTE)
    but uses the original V1 (heavily biased) dataset as input.
    Saves the resulting model to a separate directory.
    """
    print(f"--- Training Superior MultinomialNB Model on ORIGINAL BIASED Data ---")
    
    # 1. Load Original Biased Data
    print(f"Loading original data from '{ORIGINAL_DATA_PATH}'...")
    if not os.path.exists(ORIGINAL_DATA_PATH):
        print(f"❌ FATAL ERROR: Original data file not found at the specified path.")
        print("Please ensure the path is correct.")
        return
        
    df = pd.read_csv(ORIGINAL_DATA_PATH, quotechar='"', on_bad_lines='skip')
    df.dropna(subset=['Message', 'Category'], inplace=True)
    df.drop_duplicates(subset=['Message'], inplace=True)
    
    X = df["Message"].astype(str)
    y_labels = df["Category"]
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    print("Data loaded. Highlighting the heavy class imbalance:")
    print(df['Category'].value_counts())
    print("-" * 30)

    # 2. Define the Training Pipeline
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

    # 3. Train the Model
    print("Training the MultinomialNB pipeline on the biased data...")
    pipeline.fit(X, y)
    print("✅ Model training complete.")

    # 4. Save the Model and Artifacts
    print(f"Saving model artifacts to new directory: '{OUTPUT_MODEL_DIR}'")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    
    pipeline_filename = "nb_multinomial_pipeline_biased.joblib"
    label_encoder_filename = "label_encoder_biased.joblib"
    
    joblib.dump(pipeline, os.path.join(OUTPUT_MODEL_DIR, pipeline_filename))
    joblib.dump(le, os.path.join(OUTPUT_MODEL_DIR, label_encoder_filename))
    
    print("\n--- Training complete. ---")
    print(f"Model saved to '{os.path.join(OUTPUT_MODEL_DIR, pipeline_filename)}'")


if __name__ == "__main__":
    train_superior_model_on_biased_data()