# backend/database.py

import sqlite3
import pandas as pd
import os
import csv

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, 'data')

DB_PATH = os.path.join(DATA_DIR, "spamguard_feedback.db")

def init_db():
    """Initializes the feedback database and table if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True) # Ensure the data directory exists
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            label TEXT NOT NULL,
            source TEXT NOT NULL, -- 'user' or 'llm'
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def add_feedback(message: str, label: str, source: str = 'user'):
    """Adds a new feedback entry to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feedback (message, label, source) VALUES (?, ?, ?)", (message, label, source))
    conn.commit()
    conn.close()

def get_feedback_as_df():
    """Retrieves all feedback from the database as a Pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    # This query remains the same as it's independent of the CSV files
    df = pd.read_sql_query("SELECT message as Message, label as Category FROM feedback", conn)
    conn.close()
    return df

def get_all_feedback():
    """Retrieves ALL pending feedback records from the database for review."""
    conn = sqlite3.connect(DB_PATH)
    # Select all columns, including the ID, which is crucial for deletion
    df = pd.read_sql_query("SELECT id, message, label, source, timestamp FROM feedback", conn)
    conn.close()
    return df.to_dict('records') # Return as a list of dictionaries

def delete_feedback_by_ids(ids: list):
    """Deletes specific feedback records from the database by their IDs."""
    if not ids:
        return 0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Use a placeholder for each ID to prevent SQL injection
    placeholders = ','.join('?' for _ in ids)
    query = f"DELETE FROM feedback WHERE id IN ({placeholders})"
    cursor.execute(query, ids)
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count

def delete_all_feedback():
    """Deletes ALL records from the feedback table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM feedback")
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count

def get_analytics(dataset_filename: str):
    """
    Calculates and returns key analytics for a SPECIFIC dataset file.
    The filename is passed as an argument to make the function dynamic.
    """
    # Construct the full path to the specified dataset CSV
    base_dataset_path = os.path.join(DATA_DIR, dataset_filename)
    
    # Initialize base counts to 0 in case the file doesn't exist
    base_counts = {}
    if os.path.exists(base_dataset_path):
        try:
            df_base = pd.read_csv(base_dataset_path, quotechar='"', on_bad_lines='skip')
            base_counts = df_base['Category'].value_counts().to_dict()
        except Exception as e:
            print(f"Warning: Could not read analytics from {base_dataset_path}. Error: {e}")
            base_counts = {} # Reset on read error

    # The feedback data is independent of the CSV and is fetched from the DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT label, COUNT(*) FROM feedback GROUP BY label")
    feedback_counts = dict(cursor.fetchall())
    cursor.execute("SELECT source, COUNT(*) FROM feedback GROUP BY source")
    source_counts = dict(cursor.fetchall())
    conn.close()

    return {
        "base_ham_count": base_counts.get('ham', 0),
        "base_spam_count": base_counts.get('spam', 0),
        "new_ham_count": feedback_counts.get('ham', 0),
        "new_spam_count": feedback_counts.get('spam', 0),
        "user_contribution": source_counts.get('user', 0),
        "llm_contribution": source_counts.get('llm', 0)
    }

def enrich_main_dataset(dataset_filename: str) -> int:
    """
    Appends all feedback from the database to the SPECIFIED main CSV dataset,
    then clears the feedback database.
    """
    df_feedback = get_feedback_as_df()
    if df_feedback.empty:
        return 0 # No new records to add

    # Construct the full path to the target CSV file to enrich
    target_csv_path = os.path.join(DATA_DIR, dataset_filename)
    
    if not os.path.exists(target_csv_path):
        print(f"Warning: Target dataset '{target_csv_path}' for enrichment does not exist. No action taken.")
        return -1 # Return an error code

    records_to_add = df_feedback[['Category', 'Message']].values.tolist()

    try:
        # Append to the specified CSV file
        with open(target_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(records_to_add)

        # Clear the feedback database now that the data has been integrated
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM feedback")
        conn.commit()
        conn.close()
        
        print(f"Successfully enriched dataset '{dataset_filename}' with {len(df_feedback)} records.")
        return len(df_feedback)
        
    except Exception as e:
        print(f"Error during dataset enrichment for '{dataset_filename}': {e}")
        raise e