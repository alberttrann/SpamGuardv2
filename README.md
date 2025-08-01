<img width="1915" height="742" alt="image" src="https://github.com/user-attachments/assets/3ef34376-2e27-46d0-805a-7b6c7cc632e1" />


<img width="1840" height="708" alt="image" src="https://github.com/user-attachments/assets/53a73bfe-adfe-4d03-8002-2346b8a5ebe9" />


<img width="1858" height="782" alt="image" src="https://github.com/user-attachments/assets/f778d7b3-ee18-4096-85ec-bf3aa8450108" />


<img width="1557" height="381" alt="image" src="https://github.com/user-attachments/assets/a2864767-056b-4b79-a6ee-7971180b211c" />


<img width="1580" height="377" alt="image" src="https://github.com/user-attachments/assets/a781fdb8-c32d-4d79-8696-1018f5377790" />


<img width="1838" height="596" alt="image" src="https://github.com/user-attachments/assets/1a6f5c3e-36f4-4cd7-88e2-d428efe71ffa" />


<img width="1836" height="461" alt="image" src="https://github.com/user-attachments/assets/a6b1ec8b-e173-48dd-8c42-601be613cc44" />


<img width="1853" height="611" alt="image" src="https://github.com/user-attachments/assets/0344cd07-cf75-48bd-8a35-54c73f67498f" />


<img width="1446" height="1034" alt="image" src="https://github.com/user-attachments/assets/29fc2e69-66b0-4f0d-af63-30c116b38687" />


---

## README.md: Technical Analysis of Architectural Evolution

### Project: SpamGuard - An Adaptive Hybrid Classifier

This document outlines the architectural journey and technical decision-making process during the development of the SpamGuard project. The goal was to build an adaptive spam filter that combined a fast, classical machine learning model with a modern, powerful vector database for deep semantic analysis.

### I. Initial Architecture (V1): A Hybrid of Naive Bayes and Vector Search

The initial design was a two-stage hybrid classifier:

1.  **Stage 1: Fast Triage with Naive Bayes:** A Naive Bayes classifier would provide a rapid first-pass analysis of incoming messages. The design choice was to use this for efficiency, quickly dispatching obvious cases without invoking more computationally expensive models.
2.  **Stage 2: Deep Analysis with FAISS Vector Search:** For messages where the Naive Bayes classifier was uncertain, the task would be escalated to a deep learning-based vector search. This stage would use `intfloat/multilingual-e5-base` sentence embeddings to find the most semantically similar messages in a FAISS (Facebook AI Similarity Search) index and make a final prediction based on a k-Nearest Neighbors (k-NN) majority vote.

#### Initial Model Selection: `GaussianNB`

At the project's inception, with knowledge of `BernoulliNB` and `GaussianNB`, the latter was chosen. The rationale was that `GaussianNB` could potentially model the "intensity" or frequency of spam-related words, which `BernoulliNB`'s presence/absence logic would miss. The feature representation chosen was a simple Bag-of-Words (BoW) vector, where each feature is the integer count of a word's occurrence in a message.

### II. Problem Identification and Diagnosis

#### A. Data Analysis: Severe Class Imbalance

The first step was to analyze the provided dataset (`2cls_spam_text_cls_original.csv`). The analysis immediately revealed a critical issue:

**Initial Dataset Analysis (`show_data_bias.py`):**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
CSV_PATH = "C:\\Users\\alberttran\\Downloads\\2cls_spam_text_cls_original.csv"

def analyze_and_visualize_bias(file_path):
    """
    Reads a CSV file, calculates the distribution of 'ham' vs 'spam' labels,
    prints the results, and generates a visualization.
    """
    print("--- Dataset Bias Analysis ---")

    # --- 1. Check if the file exists ---
    if not os.path.exists(file_path):
        print(f"❌ ERROR: The file '{file_path}' was not found.")
        print("Please make sure the file path is correct.")
        return

    # --- 2. Load the dataset ---
    print(f"Reading data from '{file_path}'...")
    try:
        # Using parameters from your other scripts to ensure it reads correctly
        df = pd.read_csv(file_path, quotechar='"', on_bad_lines='skip')
        
        # Check if the required 'Category' column exists
        if 'Category' not in df.columns:
            print(f"❌ ERROR: The CSV file must contain a 'Category' column.")
            return

    except Exception as e:
        print(f"❌ ERROR: Failed to read or process the CSV file. Details: {e}")
        return

    # --- 3. Calculate the counts and percentages ---
    print("\nCalculating class distribution...")
    
    # Use value_counts() to get the counts of each category
    class_counts = df['Category'].value_counts()
    
    total_messages = len(df)
    
    print("\n--- Raw Counts ---")
    print(class_counts)
    
    print("\n--- Percentages ---")
    for label, count in class_counts.items():
        percentage = (count / total_messages) * 100
        print(f"  - {label.upper()}: {count} messages ({percentage:.2f}%)")
        
    print("\n--- Conclusion ---")
    print("The numbers above clearly demonstrate a significant class imbalance,")
    print("which can cause a machine learning model to be heavily biased")
    print("towards the majority class (ham).\n")
    
    # --- 4. Generate the Visualization ---
    print("Generating visualization...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    
    # Create the bar plot
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette=['#3498db', '#e74c3c'])
    
    # Add annotations (the exact number) on top of each bar
    for i, count in enumerate(class_counts.values):
        ax.text(i, count + (total_messages * 0.01), f'{count:,}', # Add a small offset for the text
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Set titles and labels for clarity
    ax.set_title('Class Distribution in the Dataset', fontsize=16, fontweight='bold')
    ax.set_xlabel('Message Category', fontsize=12)
    ax.set_ylabel('Number of Messages', fontsize=12)
    ax.set_ylim(0, total_messages * 1.1) # Give some space at the top

    # Display the plot
    plt.show()


if __name__ == "__main__":
    analyze_and_visualize_bias(CSV_PATH)
```

```
--- Raw Counts ---
Category
ham     4825
spam     747
Name: count, dtype: int64

--- Percentages ---
  - HAM: 4825 messages (86.59%)
  - SPAM: 747 messages (13.41%)
```

A dataset with an 87/13 split presents a significant risk. Any classifier trained on this data would develop a strong prior probability towards the majority class (`ham`), potentially leading to poor performance in identifying the minority class (`spam`), which is often the primary target in filtering systems.

#### B. First Mitigation Attempt: LLM-based Data Augmentation

To address the class imbalance, the first architectural enhancement was the integration of a data augmentation module powered by Large Language Models (LLMs). This module was designed to generate high-quality, synthetic spam messages, thereby increasing the representation of the minority class.

**Post-Augmentation Dataset Analysis:**
```
--- Raw Counts ---
Category
ham     4881
spam    2257
Name: count, dtype: int64

--- Percentages ---
  - HAM: 4881 messages (68.38%)
  - SPAM: 2257 messages (31.62%)
```
This successfully improved the class ratio from approximately 6.5:1 to 2.2:1, creating a more balanced training environment.

#### C. Performance Evaluation and Anomaly Detection

Following the data augmentation, the `GaussianNB` model was retrained on the new, more balanced dataset. An evaluation was conducted on a hold-out test set (`evaluation_data.txt`). To establish a baseline, a second experiment was run where a temporary model was trained on the *original biased data* and evaluated on the same test set.

**Evaluation Results (Model trained on Augmented Data):**
```python
# evaluate_old_model.py

import joblib
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
MODEL_DIR = "models"
NB_MODEL_PATH = os.path.join(MODEL_DIR, "nb_model.joblib")
DICTIONARY_PATH = os.path.join(MODEL_DIR, "dictionary.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
TEST_DATA_PATH = "evaluation_data.txt" 

# --- Preprocessing Logic  ---

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK punkt tokenizer not found. Downloading...")
    nltk.download('punkt')

stemmer = PorterStemmer()

def preprocess_text(text: str) -> list:
    """Preprocessing steps for the old Naive Bayes model."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def create_features(tokens: list, dictionary: list) -> np.ndarray:
    """Creates a Bag-of-Words feature vector for the old Naive Bayes."""
    if not isinstance(dictionary, list):
        raise TypeError("Dictionary must be a list.")
        
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            # Find the index of the token and increment the count
            features[dictionary.index(token)] += 1
    return features

def evaluate_model():
    """Main function to load the model, run evaluation, and print results."""
    print("--- Starting Evaluation of OLD GaussianNB Model ---")

    # --- 1. Load Model Artifacts ---
    print("Loading model, dictionary, and label encoder...")
    try:
        nb_model = joblib.load(NB_MODEL_PATH)
        dictionary = joblib.load(DICTIONARY_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print("✅ Artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find a model file: {e}")
        print("Please ensure you have the 'nb_model.joblib', 'dictionary.joblib', and 'label_encoder.joblib' files in the 'models' directory.")
        return

    # --- 2. Load and Parse Test Data ---
    print(f"Loading test data from '{TEST_DATA_PATH}'...")
    try:
        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ ERROR: Test data file not found at '{TEST_DATA_PATH}'.")
        print("Please create this file and format it as 'label,message' per line.")
        return

    true_labels = []
    messages = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            label, message = line.split(',', 1)
            if label not in ['ham', 'spam']:
                print(f"Warning: Skipping line {i+1} due to invalid label: '{label}'")
                continue
            true_labels.append(label)
            messages.append(message)
        except ValueError:
            print(f"Warning: Skipping line {i+1} due to incorrect format. Expected 'label,message'.")
            continue
    
    if not messages:
        print("❌ ERROR: No valid data could be loaded from the test file.")
        return
        
    print(f"✅ Loaded {len(messages)} messages for evaluation.")

    # --- 3. Make Predictions ---
    print("Making predictions on the test set...")
    predictions = []
    for msg in messages:
        processed_text = preprocess_text(msg)
        features = create_features(processed_text, dictionary)
        features_reshaped = np.array(features).reshape(1, -1)
        
        # Predict the numeric label (e.g., [0] or [1])
        numeric_prediction = nb_model.predict(features_reshaped)
        
        # Convert the numeric label back to a string ('ham' or 'spam')
        string_prediction = label_encoder.inverse_transform(numeric_prediction)
        
        predictions.append(string_prediction[0])
    print("✅ Predictions complete.")

    # --- 4. Calculate and Display Metrics ---
    print("\n--- Evaluation Results ---")
    
    # Accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print("Accuracy is the percentage of total correct predictions.")

    # Classification Report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    report = classification_report(true_labels, predictions, target_names=label_encoder.classes_)
    print(report)
    

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions, labels=label_encoder.classes_)
    print(cm)
    
    # --- 5. Visualize the Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix for Old GaussianNB Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # --- 6. Show Misclassified Examples ---
    print("\n--- Examples of Misclassifications ---")
    misclassified_count = 0
    for i in range(len(messages)):
        if true_labels[i] != predictions[i]:
            misclassified_count += 1
            print(f"\nMessage #{i+1}:")
            print(f"  - Text: \"{messages[i]}\"")
            print(f"  - Actual Label:   {true_labels[i].upper()}")
            print(f"  - Predicted as:   {predictions[i].upper()}   <--- WRONG")
    
    if misclassified_count == 0:
        print("\n🎉 No misclassifications found on this test set!")

if __name__ == "__main__":
    evaluate_model()
```
```
Overall Accuracy: 59.78%
Confusion Matrix:
[[27 19]  <- (True Ham, Predicted Spam)
 [18 28]]  <- (True Spam, Predicted Ham)
```

**Evaluation Results (Model trained on Original Biased Data):**
```python
# train_and_evaluate_original.py

import os
import shutil
import string

import faiss
import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# --- Configuration ---
# Path to the original, raw dataset 
ORIGINAL_DATA_PATH = r"C:\Users\alberttran\Downloads\2cls_spam_text_cls_original.csv"
# Path to evaluation file
EVALUATION_DATA_PATH = "evaluation_data.txt"
# A temporary directory to store the newly trained model to avoid conflicts
TEMP_MODEL_DIR = "temp_original_model"


# --- NLTK Setup ---
def setup_nltk():
    """Downloads necessary NLTK packages if they don't exist."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("NLTK 'stopwords' not found. Downloading...")
        nltk.download('stopwords', quiet=True)
setup_nltk()

# --- Part 1: Training Logic (Replicating the old `train_nb.py`) ---

# Preprocessing logic match the old model's logic .
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def create_dictionary(messages):
    all_words = set(word for message in messages for word in message)
    return sorted(list(all_words))

def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1
    return features

def train_original_model():
    """Trains a new GaussianNB model from the original dataset."""
    print("--- Part 1: Training New Model on Original Biased Dataset ---")
    
    # 1. Load Data
    print(f"Loading original data from '{ORIGINAL_DATA_PATH}'...")
    if not os.path.exists(ORIGINAL_DATA_PATH):
        print(f"❌ FATAL ERROR: Original data file not found at the specified path.")
        return False
        
    df = pd.read_csv(ORIGINAL_DATA_PATH, quotechar='"', on_bad_lines='skip')
    df.dropna(subset=['Message', 'Category'], inplace=True)
    
    messages = df["Message"].astype(str).values.tolist()
    labels = df["Category"].values.tolist()
    
    print(f"Data loaded. Distribution: {df['Category'].value_counts().to_dict()}")

    # 2. Preprocess Text
    print("Preprocessing text data...")
    messages_processed = [preprocess_text(message) for message in messages]

    # 3. Feature Extraction (Bag-of-Words)
    print("Creating dictionary and feature vectors...")
    dictionary = create_dictionary(messages_processed)
    X = np.array([create_features(msg, dictionary) for msg in messages_processed])

    # 4. Process Labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # 5. Train GaussianNB Model
    print("Training GaussianNB model...")
    model = GaussianNB()
    model.fit(X, y)

    # 6. Save the Model and Artifacts to the temporary directory
    print(f"Saving temporary model to '{TEMP_MODEL_DIR}'...")
    os.makedirs(TEMP_MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(TEMP_MODEL_DIR, 'nb_model.joblib'))
    joblib.dump(dictionary, os.path.join(TEMP_MODEL_DIR, 'dictionary.joblib'))
    joblib.dump(le, os.path.join(TEMP_MODEL_DIR, 'label_encoder.joblib'))
    print("✅ Training complete.")
    return True


# --- Part 2: Evaluation Logic (Replicating the app's `SpamGuardClassifier`) ---

# Helper functions for the vector search part
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embeddings(texts, model, tokenizer, device, prefix, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({prefix})"):
        batch_texts = [f"{prefix}: {text}" for text in texts[i:i + batch_size]]
        batch_dict = tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


def evaluate_hybrid_classifier():
    """Evaluates the full hybrid classifier logic using the temporary model."""
    print("\n--- Part 2: Evaluating the Hybrid Classifier ---")

    # 1. Load the TEMPORARY Naive Bayes model components
    print(f"Loading temporary model from '{TEMP_MODEL_DIR}'...")
    nb_model = joblib.load(os.path.join(TEMP_MODEL_DIR, 'nb_model.joblib'))
    dictionary = joblib.load(os.path.join(TEMP_MODEL_DIR, 'dictionary.joblib'))
    label_encoder = joblib.load(os.path.join(TEMP_MODEL_DIR, 'label_encoder.joblib'))

    # 2. Setup the Vector Search components
    print("Setting up Vector Search database...")
    # Load the Transformer model
    MODEL_NAME = "intfloat/multilingual-e5-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    transformer_model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"Transformer model loaded on {device}.")

    # Load original data AGAIN for the FAISS index
    df_orig = pd.read_csv(ORIGINAL_DATA_PATH, quotechar='"', on_bad_lines='skip')
    df_orig.dropna(subset=['Message'], inplace=True)
    all_messages_orig = df_orig["Message"].astype(str).tolist()
    all_labels_orig = df_orig["Category"].tolist()
    
    # Create embeddings and FAISS index
    passage_embeddings = get_embeddings(all_messages_orig, transformer_model, tokenizer, device, "passage")
    embedding_dim = passage_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(passage_embeddings.astype('float32'))
    print("✅ FAISS index built successfully from original data.")

    # 3. Load Evaluation Data
    print(f"Loading evaluation data from '{EVALUATION_DATA_PATH}'...")
    if not os.path.exists(EVALUATION_DATA_PATH):
        print(f"❌ FATAL ERROR: Evaluation data file not found at '{EVALUATION_DATA_PATH}'.")
        return
    with open(EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    true_labels, eval_messages = [], []
    for line in lines:
        if line.strip():
            label, message = line.split(',', 1)
            true_labels.append(label.strip())
            eval_messages.append(message.strip())
    print(f"✅ Loaded {len(eval_messages)} messages for evaluation.")

    # 4. Run the full classification logic for each message
    print("Running evaluation loop...")
    predictions = []
    for msg in tqdm(eval_messages, desc="Classifying"):
        # Stage 1: Fast Triage with Naive Bayes
        processed_text_nb = preprocess_text(msg)
        features_nb = create_features(processed_text_nb, dictionary)
        features_nb = np.array(features_nb).reshape(1, -1)
        
        nb_probabilities = nb_model.predict_proba(features_nb)[0]
        spam_class_index = np.where(label_encoder.classes_ == 'spam')[0][0]
        spam_prob = nb_probabilities[spam_class_index]
        
        prediction = None
        if spam_prob < 0.1:
            prediction = "ham"
        elif spam_prob > 0.9:
            prediction = "spam"
        else:
            # Stage 2: Deep Analysis with Vector Search
            query_embedding = get_embeddings([msg], transformer_model, tokenizer, device, "query", batch_size=1)
            scores, indices = faiss_index.search(query_embedding.astype('float32'), k=5)
            neighbor_labels = [all_labels_orig[i] for i in indices[0]]
            prediction = max(set(neighbor_labels), key=neighbor_labels.count)
        
        predictions.append(prediction)

    # 5. Report Results
    print("\n--- Evaluation Results (Original Model on Biased Data) ---")
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    report = classification_report(true_labels, predictions, target_names=label_encoder.classes_, zero_division=0)
    print(report)
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions, labels=label_encoder.classes_)
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Model Trained on Original Biased Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


if __name__ == "__main__":
    # Ensure the temp directory doesn't exist from a failed previous run
    if os.path.exists(TEMP_MODEL_DIR):
        shutil.rmtree(TEMP_MODEL_DIR)

    try:
        # Step 1: Train the new model
        training_successful = train_original_model()
        
        # Step 2: Evaluate if training was successful
        if training_successful:
            evaluate_hybrid_classifier()
            
    finally:
        # Step 3: Clean up the temporary directory
        if os.path.exists(TEMP_MODEL_DIR):
            print(f"\nCleaning up temporary directory: '{TEMP_MODEL_DIR}'")
            shutil.rmtree(TEMP_MODEL_DIR)
            print("✅ Cleanup complete.")
```

```
Overall Accuracy: 59.78%
Confusion Matrix:
[[27 19]
 [18 28]]
```
The results were startling: they were **absolutely identical**. This was a critical finding. The fact that mitigating the severe data bias had **zero impact** on the final evaluation metrics pointed to a fundamental architectural flaw that went deeper than the dataset itself.

The logical conclusion was that the **Stage 2 Vector Search was never being activated**. The `GaussianNB` triage model, regardless of its training data, was always making a high-confidence prediction (`P(spam) < 0.1` or `P(spam) > 0.9`), thereby never flagging a message as "uncertain." The system was not functioning as a hybrid; it was only ever a `GaussianNB` classifier.

### III. The Architectural Pivot: From `GaussianNB` to `MultinomialNB`

The identical evaluation results forced a re-examination of the initial model choice. The core problem was a mathematical mismatch between the model and the data.

*   **The Flaw of `GaussianNB`:** `GaussianNB` computes the likelihood `P(x_i | y)` by assuming the feature values `x_i` (word counts) are sampled from a continuous Gaussian distribution. This assumption is fundamentally incorrect for text data, where word counts are discrete, non-negative integers, and the data distribution is sparse and highly skewed towards zero. This mismatch led to systemically overconfident and unreliable probability estimates.

*   **The Correct Tool: `MultinomialNB`:** Upon further study, it became clear that `MultinomialNB` is the mathematically appropriate Naive Bayes variant for this task. It is specifically designed for features that represent discrete counts or frequencies. Its likelihood function `P(x_i | y)` is calculated based on the frequency of word `x_i` in class `y` across the entire training corpus, which directly aligns with the nature of Bag-of-Words and TF-IDF feature vectors.

This insight prompted a crucial architectural pivot.

### IV. Final Architecture (V2): An Optimized and Functional Hybrid System

The final, superior architecture incorporates the right tool for the job, making the system both simpler and more powerful.

1.  **Stage 1: Fast Triage with `MultinomialNB`**
    *   **Model:** `GaussianNB` is replaced with `MultinomialNB`.
    *   **Feature Engineering:** The simple BoW model is replaced with a `TfidfVectorizer` using a 1-2 N-gram range. This provides a richer feature set that captures word importance and local context.
    *   **Data Balancing:** To further enhance robustness, `SMOTE` (Synthetic Minority Over-sampling Technique) is integrated into the training pipeline. This ensures that even within the `MultinomialNB` framework, the model is trained on a perfectly balanced dataset, forcing it to learn the distinguishing features of spam without bias.

2.  **Stage 2: Deep Analysis with FAISS Vector Search (Unchanged)**
    *   This component remains unchanged, but it is now a functional part of the system. The more reliable and less confident probability estimates from `MultinomialNB` now allow the triage stage to correctly identify and escalate ambiguous cases.

This architectural evolution from a flawed `GaussianNB` implementation to a correctly specified `MultinomialNB` pipeline represents the key learning of this project. It highlights the critical importance of selecting a model whose mathematical assumptions align with the intrinsic properties of the data, a principle that proved to be even more impactful than mitigating data imbalance alone.
