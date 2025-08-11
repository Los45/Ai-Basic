#author: Naufan
#description: Chatbot sederhana dengan pembelajaran berbasis semantik
#date: 2025-8-10
import json
import pickle
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

DATA_FILE = Path("intents.json")
EMBED_FILE = Path("embeddings.pkl")
MODEL_NAME_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.6  # semakin tinggi → lebih ketat

# ============ Load dataset & embeddings =============
if not DATA_FILE.exists():
    raise FileNotFoundError("intents.json tidak ditemukan")

with DATA_FILE.open(encoding="utf-8") as fp:
    intents_json = json.load(fp)

if not EMBED_FILE.exists():
    print("Embeddings belum ada. Jalankan train.py dulu.")
    exit()

with EMBED_FILE.open("rb") as fp:
    emb_data = pickle.load(fp)

# Pastikan model tersedia
encoder = SentenceTransformer(emb_data.get("model_name", MODEL_NAME_FALLBACK))

patterns = emb_data["patterns"]
pattern_tags = emb_data["tags"]
embeddings = emb_data["embeddings"]  # numpy array

# Buat map tag → responses
RESPONSES = {i["tag"]: i["responses"] for i in intents_json["intents"]}

# ====================================================
print("🤖 Chatbot semantic siap! Ketik 'bye' untuk keluar.\n")

while True:
    user_input = input("Kamu: ").strip()
    if user_input.lower() in {"bye", "exit", "keluar"}:
        print("Bot: Sampai jumpa! 👋")
        break

    query_vec = encoder.encode(user_input)
    cos_scores = util.cos_sim(query_vec, embeddings)[0]
    best_idx = int(cos_scores.argmax())
    best_score = float(cos_scores[best_idx])

    if best_score >= SIM_THRESHOLD:
        tag = pattern_tags[best_idx]
        print("Bot:", random.choice(RESPONSES[tag]))
        continue

    # —— Bot tidak yakin ——
    print("Bot: Maaf, saya belum mengerti. Boleh jelaskan maksud pertanyaan?")
    teach = input("➤ Jelaskan (atau ketik 'skip'): ").strip()
    if teach.lower() == "skip":
        continue

    # Proses belajar: minta tag & jawaban ideal
    new_tag = input("➤ Tag/topik untuk pertanyaan ini?: ").strip()
    new_resp = input("➤ Jawaban ideal?: ").strip()

    # Tambahkan ke intents_json (update memori)
    for intent in intents_json["intents"]:
        if intent["tag"] == new_tag:
            intent["patterns"].append(user_input)
            intent["responses"].append(new_resp)
            break
    else:
        intents_json["intents"].append({
            "tag": new_tag,
            "patterns": [user_input],
            "responses": [new_resp]
        })

    # Simpan intents.json
    DATA_FILE.write_text(json.dumps(intents_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print("🧠 Disimpan di intents.json & memory lokal.")

    # Tambah embedding baru (tanpa retrain penuh)
    patterns.append(user_input)
    pattern_tags.append(new_tag)
    embeddings = util.tensor_or_numpy(embeddings)
    new_emb = encoder.encode(user_input)
    embeddings = util.tensor_or_numpy(embeddings)
    embeddings = util.torch.cat([util.torch.tensor(embeddings), util.torch.tensor(new_emb).unsqueeze(0)], dim=0)

    # Simpan embeddings baru
    with EMBED_FILE.open("wb") as fp:
        pickle.dump({"model_name": MODEL_NAME_FALLBACK, "embeddings": embeddings.numpy(), "patterns": patterns, "tags": pattern_tags}, fp)

    # Muat ulang embeddings ke memori (numpy array)
    embeddings = embeddings.numpy()

    # Tambah respons ke dict jika tag baru
    RESPONSES.setdefault(new_tag, []).append(new_resp)

    print("🤖 Berhasil belajar! Coba pertanyaan tadi lagi.\n")