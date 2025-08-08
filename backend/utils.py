# backend/utils.py

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()
english_stop_words = set(stopwords.words('english'))

def preprocess_tokenizer(text: str) -> list:
    """
    This is the official, shared tokenizer for the SpamGuard project.
    It handles all text cleaning, tokenization, stop word removal, and stemming.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    
    filtered_and_stemmed_tokens = [
        stemmer.stem(token) for token in tokens if token.isalpha() and token not in english_stop_words
    ]
    return filtered_and_stemmed_tokens