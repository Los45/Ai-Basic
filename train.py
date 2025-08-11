# === train.py ===
"""Pre‑compute sentence embeddings for each pattern using a
   pre‑trained transformer (semantic matching). No more traditional
   classifier –bot akan "paham" makna kalimat.
   Jalankan sekali:  python train.py
"""

import json
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ringan & cepat
DATA_FILE = Path("intents.json")
EMBED_FILE = Path("embeddings.pkl")


def load_intents():
    if not DATA_FILE.exists():
        raise FileNotFoundError("intents.json tidak ditemukan")
    with DATA_FILE.open(encoding="utf-8") as fp:
        return json.load(fp)


def main():
    intents_json = load_intents()

    # Flatten patterns → list, dan simpan (index ↔ tag)
    patterns, tags = [], []
    for intent in intents_json["intents"]:
        tag = intent["tag"]
        for p in intent["patterns"]:
            patterns.append(p)
            tags.append(tag)

    # Encode patterns menjadi vektor 384‑D
    encoder = SentenceTransformer(MODEL_NAME)
    embeddings = encoder.encode(patterns, show_progress_bar=True)

    # Simpan ke pickle
    with EMBED_FILE.open("wb") as fp:
        pickle.dump({"model_name": MODEL_NAME, "embeddings": embeddings, "patterns": patterns, "tags": tags}, fp)

    print("✅ Embedding selesai & disimpan →", EMBED_FILE)


if __name__ == "__main__":
    main()
