import os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "models/neomind_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    # Simple initial model
    return {
        "vectorizer": TfidfVectorizer(),
        "classifier": LogisticRegression(),
        "trained": False
    }

def save_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
