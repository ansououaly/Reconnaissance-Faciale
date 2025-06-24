import os
import torch
import pickle

DB_PATH = "data/face_db.pkl"

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_to_db(name, embedding):
    db = load_db()
    db[name] = embedding
    os.makedirs("data", exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)