# backend/loader.py

import time
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.classifier import SpamGuardClassifier

FLAG_FILE_PATH = os.path.join(os.path.dirname(__file__), "_ready.flag")

def main():
    """
    This script's sole purpose is to load the classifier and create a flag file
    to signal that the main server can now use it safely.
    """
    print("--- Starting SpamGuard Model Loader ---")
    
    # Remove any old flag file to signal we are starting a fresh load
    if os.path.exists(FLAG_FILE_PATH):
        os.remove(FLAG_FILE_PATH)
        print("Removed old ready flag.")

    # This creates the lazy instance
    classifier = SpamGuardClassifier()
    
    # This triggers the long, blocking load process.
    classifier.load() 
    
    with open(FLAG_FILE_PATH, "w") as f:
        f.write("ready")
    
    print("\n--- SpamGuard Models are LOADED and READY. ---")
    print("The main server can now accept classification requests.")

if __name__ == "__main__":
    main()