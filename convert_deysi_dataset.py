# convert_deysi_dataset.py

import os
from datasets import load_dataset
from tqdm import tqdm
import csv

# --- Configuration ---
HF_DATASET_ID = "Deysi/spam-detection-dataset"
TRAIN_OUTPUT_FILENAME = "deysi_train.txt"
TEST_OUTPUT_FILENAME = "deysi_test.txt"


def process_and_write_split(split_name: str, output_filename: str):
    """
    Downloads a specific split of the dataset, processes it, and writes
    it to a local text file in the 'label,message' format.
    """
    print("-" * 60)
    print(f"Processing '{split_name}' split from '{HF_DATASET_ID}'...")
    
    try:
        # 1. Load the specific split from Hugging Face
        dataset = load_dataset(HF_DATASET_ID, split=split_name)
        
        # 2. Filter out any rows with missing or empty data
        dataset = dataset.filter(
            lambda x: x['label'] is not None and x['text'] is not None and str(x['text']).strip() != ""
        )
        
        print(f"Found {len(dataset)} valid rows to process.")
        
        # 3. Open the output file and process the data
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            # Using csv.writer to correctly handle any commas or quotes within the message text
            writer = csv.writer(f)
            
            for item in tqdm(dataset, desc=f"Writing to {output_filename}"):
                # Extract the required columns
                message = item['text']
                label_raw = item['label']
                
                # Map 'not_spam' to 'ham' to match SpamGuard's format
                label_final = "ham" if label_raw == "not_spam" else "spam"
                
                # Write the row in the desired format
                writer.writerow([label_final, message])
                
        print(f"✅ Successfully created '{output_filename}' with {len(dataset)} records.")

    except Exception as e:
        print(f"❌ An error occurred while processing the '{split_name}' split: {e}")


if __name__ == "__main__":
    # Process both the train and test splits
    process_and_write_split('train', TRAIN_OUTPUT_FILENAME)
    process_and_write_split('test', TEST_OUTPUT_FILENAME)
    
    print("\n" + "="*60)
    print("All processing complete.")
    print(f"'{TRAIN_OUTPUT_FILENAME}' and '{TEST_OUTPUT_FILENAME}' are now ready in your project directory.")