<img width="1915" height="742" alt="image" src="https://github.com/user-attachments/assets/3ef34376-2e27-46d0-805a-7b6c7cc632e1" />


<img width="1840" height="708" alt="image" src="https://github.com/user-attachments/assets/53a73bfe-adfe-4d03-8002-2346b8a5ebe9" />


<img width="1858" height="782" alt="image" src="https://github.com/user-attachments/assets/f778d7b3-ee18-4096-85ec-bf3aa8450108" />


<img width="1557" height="381" alt="image" src="https://github.com/user-attachments/assets/a2864767-056b-4b79-a6ee-7971180b211c" />


<img width="1580" height="377" alt="image" src="https://github.com/user-attachments/assets/a781fdb8-c32d-4d79-8696-1018f5377790" />


<img width="1838" height="596" alt="image" src="https://github.com/user-attachments/assets/1a6f5c3e-36f4-4cd7-88e2-d428efe71ffa" />


<img width="1836" height="461" alt="image" src="https://github.com/user-attachments/assets/a6b1ec8b-e173-48dd-8c42-601be613cc44" />


<img width="1853" height="611" alt="image" src="https://github.com/user-attachments/assets/0344cd07-cf75-48bd-8a35-54c73f67498f" />


<img width="1446" height="1034" alt="image" src="https://github.com/user-attachments/assets/29fc2e69-66b0-4f0d-af63-30c116b38687" />


The original data is heavily biased towards "ham":
```python
# show_data_bias.py

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
result:
```
PS E:\AIO\conquer_project> python show_data_bias.py
--- Dataset Bias Analysis ---
Reading data from 'C:\Users\alberttran\Downloads\2cls_spam_text_cls_original.csv'...

Calculating class distribution...

--- Raw Counts ---
Category
ham     4825
spam     747
Name: count, dtype: int64

--- Percentages ---
  - HAM: 4825 messages (86.59%)
  - SPAM: 747 messages (13.41%)

--- Conclusion ---
The numbers above clearly demonstrate a significant class imbalance,
which can cause a machine learning model to be heavily biased
towards the majority class (ham).
```

That's the reason why I decided to do data augmentation with LLM, which improve the bias scale a lil bit. The .csv file in this repo is the post-augmentation version:
```
PS E:\AIO\conquer_project> python show_data_bias.py
--- Dataset Bias Analysis ---
Reading data from 'E:\AIO\conquer_project\backend\data\2cls_spam_text_cls.csv'...

Calculating class distribution...

--- Raw Counts ---
Category
ham     4881
spam    2257
Name: count, dtype: int64

--- Percentages ---
  - HAM: 4881 messages (68.38%)
  - SPAM: 2257 messages (31.62%)
```
You can find the original file here: https://drive.google.com/file/d/1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R/view

After using LLM to generate synthetic data to mitigate data bias, I retrain the model, and then do the evaluation:
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
    # Ensure dictionary is a list for .index() to work
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
result:
```
(.venv) PS E:\AIO\conquer_project> python evaluate_old_model.py
>>
--- Starting Evaluation of OLD GaussianNB Model ---
Loading model, dictionary, and label encoder...
✅ Artifacts loaded successfully.
Loading test data from 'evaluation_data.txt'...
✅ Loaded 92 messages for evaluation.
Making predictions on the test set...
✅ Predictions complete.

--- Evaluation Results ---

Overall Accuracy: 59.78%
Accuracy is the percentage of total correct predictions.

Classification Report:
              precision    recall  f1-score   support

         ham       0.60      0.59      0.59        46
        spam       0.60      0.61      0.60        46


         ham       0.60      0.59      0.59        46
        spam       0.60      0.61      0.60        46

    accuracy                           0.60        92
   macro avg       0.60      0.60      0.60        92
weighted avg       0.60      0.60      0.60        92


Confusion Matrix:
[[27 19]
 [18 28]]
```

which is much better than the original version on original data:


