# convert_all_datasets.py

import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import csv

# --- Configuration ---
# Hugging Face Dataset IDs
ENRON_HF_ID = "SetFit/enron_spam"
TELEGRAM_HF_ID = "thehamkercat/telegram-spam-ham"

# Local CSV Filename
KAGGLE_CSV_PATH = "email_spam.csv"

# --- Output Filenames ---
ENRON_TRAIN_OUTPUT = "enron_train.txt"
ENRON_TEST_OUTPUT = "enron_test.txt"
TELEGRAM_TEST_OUTPUT = "telegram_test.txt"
KAGGLE_TEST_OUTPUT = "kaggle_test.txt"


def process_enron_dataset():
    """
    Downloads the Enron dataset, processes both train and test splits,
    and saves them to local .txt files.
    """
    print("\n" + "="*60)
    print(f"Processing Hugging Face Dataset: {ENRON_HF_ID}")
    print("="*60)
    
    for split_name, output_filename in [('train', ENRON_TRAIN_OUTPUT), ('test', ENRON_TEST_OUTPUT)]:
        try:
            print(f"\n- Processing '{split_name}' split...")
            dataset = load_dataset(ENRON_HF_ID, split=split_name)
            
            dataset = dataset.filter(
                lambda x: x['label_text'] is not None and x['message'] is not None and str(x['message']).strip() != ""
            )
            print(f"Found {len(dataset)} valid rows to process.")

            with open(output_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for item in tqdm(dataset, desc=f"Writing {output_filename}"):
                    # The label is already 'ham' or 'spam', which is perfect.
                    label = item['label_text']
                    message = item['message']
                    writer.writerow([label, message])
            
            print(f"✅ Successfully created '{output_filename}'.")

        except Exception as e:
            print(f"❌ An error occurred while processing the Enron '{split_name}' split: {e}")

def process_telegram_dataset():
    """
    Downloads the Telegram dataset and creates a balanced 2000-sample
    subset for testing, saving it to a local .txt file.
    """
    print("\n" + "="*60)
    print(f"Processing Hugging Face Dataset: {TELEGRAM_HF_ID}")
    print("="*60)
    
    try:
        print("- Loading 'train' split and creating a balanced 2k sample subset...")
        dataset = load_dataset(TELEGRAM_HF_ID, split='train')
        
        dataset = dataset.filter(
            lambda x: x['text_type'] is not None and x['text'] is not None and str(x['text']).strip() != ""
        )
        
        # Create the balanced subset as requested
        ham_data = dataset.filter(lambda x: x['text_type'] == "ham").shuffle(seed=42).select(range(1000))
        spam_data = dataset.filter(lambda x: x['text_type'] == "spam").shuffle(seed=42).select(range(1000))
        final_dataset = concatenate_datasets([ham_data, spam_data]).shuffle(seed=42)
        
        print(f"Subset created with {len(final_dataset)} rows (1000 ham, 1000 spam).")

        with open(TELEGRAM_TEST_OUTPUT, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in tqdm(final_dataset, desc=f"Writing {TELEGRAM_TEST_OUTPUT}"):
                # The label is 'text_type' and content is 'text'
                label = item['text_type']
                message = item['text']
                writer.writerow([label, message])
        
        print(f"✅ Successfully created '{TELEGRAM_TEST_OUTPUT}'.")

    except Exception as e:
        print(f"❌ An error occurred while processing the Telegram dataset: {e}")

def process_kaggle_csv():
    """
    Loads the local email_spam.csv, processes it, and saves it
    to a standardized .txt file format.
    """
    print("\n" + "="*60)
    print(f"Processing Local CSV Dataset: {KAGGLE_CSV_PATH}")
    print("="*60)
    
    if not os.path.exists(KAGGLE_CSV_PATH):
        print(f"❌ ERROR: File not found at '{KAGGLE_CSV_PATH}'. Please make sure it's in the project root directory.")
        return

    try:
        df = pd.read_csv(KAGGLE_CSV_PATH)
        df.dropna(subset=['text', 'type'], inplace=True)
        
        print(f"Found {len(df)} valid rows to process.")

        with open(KAGGLE_TEST_OUTPUT, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Writing {KAGGLE_TEST_OUTPUT}"):
                message = str(row['text'])
                label_raw = str(row['type']).lower()
                
                # Map 'not spam' to 'ham'
                label_final = "ham" if label_raw == "not spam" else "spam"
                
                writer.writerow([label_final, message])
        
        print(f"✅ Successfully created '{KAGGLE_TEST_OUTPUT}'.")

    except Exception as e:
        print(f"❌ An error occurred while processing the Kaggle CSV: {e}")


if __name__ == "__main__":
    process_enron_dataset()
    process_telegram_dataset()
    process_kaggle_csv()
    
    print("\n" + "="*60)
    print("All data conversion tasks are complete.")
    print("The following files are now ready in your project directory:")
    print(f"- {ENRON_TRAIN_OUTPUT}")
    print(f"- {ENRON_TEST_OUTPUT}")
    print(f"- {TELEGRAM_TEST_OUTPUT}")
    print(f"- {KAGGLE_TEST_OUTPUT}")