import sqlite3
import pandas as pd
import os
import csv

DB_PATH = os.path.join("data", "spamguard_feedback.db")
CSV_PATH = os.path.join("data", "2cls_spam_text_cls.csv")

def init_db():
    """Initializes the feedback database and table if they don't exist."""
    os.makedirs("data", exist_ok=True)
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
    df = pd.read_sql_query("SELECT message as Message, label as Category FROM feedback", conn)
    conn.close()
    return df

def get_analytics():
    """Calculates and returns key analytics."""
    # Base dataset stats
    df_base = pd.read_csv(CSV_PATH, quotechar='"', on_bad_lines='skip')
    base_counts = df_base['Category'].value_counts().to_dict()

    # Feedback stats
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

def enrich_main_dataset():
    """
    Appends all feedback to the main CSV dataset using the csv module
    to ensure proper quoting and prevent parsing errors.
    """
    df_feedback = get_feedback_as_df()
    if df_feedback.empty:
        return 0

    # Get the list of new records
    # df_feedback has columns 'Message' and 'Category'
    # We want to write 'Category', 'Message' to the CSV
    records_to_add = df_feedback[['Category', 'Message']].values.tolist()

    try:
        # Open the CSV in append mode, using the csv writer
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerows(records_to_add)

        # Clear feedback DB only after a successful write
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM feedback")
        conn.commit()
        conn.close()
        
        print(f"Successfully enriched dataset with {len(df_feedback)} records.")
        return len(df_feedback)
        
    except Exception as e:
        print(f"Error during dataset enrichment: {e}")
        raise e