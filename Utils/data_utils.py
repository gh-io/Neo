import json, os

DATA_PATH = "data/training_data.jsonl"
PROCESSED_PATH = "data/processed_ids.json"

def append_data(data):
    """Append new data entries to dataset"""
    with open(DATA_PATH, "a") as f:
        for d in data if isinstance(data, list) else [data]:
            if "id" not in d:
                d["id"] = str(hash(d["text"]))  # simple unique id
            json.dump(d, f)
            f.write("\n")

def load_data():
    """Load all data"""
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, "r") as f:
        return [json.loads(line) for line in f]

def load_new_data():
    """Load only new data that hasn't been trained on"""
    all_data = load_data()
    processed_ids = set()
    if os.path.exists(PROCESSED_PATH):
        with open(PROCESSED_PATH, "r") as f:
            processed_ids = set(json.load(f))
    new_data = [d for d in all_data if d["id"] not in processed_ids]
    return new_data

def mark_processed(data):
    """Mark data as processed after training"""
    processed_ids = set()
    if os.path.exists(PROCESSED_PATH):
        with open(PROCESSED_PATH, "r") as f:
            processed_ids = set(json.load(f))
    processed_ids.update([d["id"] for d in data])
    with open(PROCESSED_PATH, "w") as f:
        json.dump(list(processed_ids), f)
