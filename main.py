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

    # Update embeddings
    if new_tag not in pattern_tags:   
        pattern_tags.append(new_tag)
        embeddings = util.torch.cat([util.torch.tensor(embeddings), util.torch.tensor(encoder.encode(user_input)).unsqueeze(0)], dim=0)

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


    # Simpan perubahan ke file embeddings
    with EMBED_FILE.open("wb") as fp:
        pickle.dump({"model_name": MODEL_NAME_FALLBACK, "embeddings": embeddings, "patterns": patterns, "tags": pattern_tags}, fp)

    print("✅ Embedding selesai & disimpan →", EMBED_FILE)
    print("🧠 Disimpan di intents.json & memory lokal.")
    print("🤖 Berhasil belajar! Coba pertanyaan tadi lagi.\n")


    #admin: Tambah fungsi untuk menghapus tag
    def delete_tag(tag):
        """Hapus tag dan semua pola terkait dari memori."""
        if tag in pattern_tags:
            idx = pattern_tags.index(tag)
            del pattern_tags[idx]
            del patterns[idx]
            del embeddings[idx]
            RESPONSES.pop(tag, None)
            print(f"Tag '{tag}' berhasil dihapus.")
        else:
            print(f"Tag '{tag}' tidak ditemukan dalam memori.")

    def list_tags():
        """Tampilkan semua tag yang ada."""
        if pattern_tags:
            print("Daftar tag yang tersedia:")
            for tag in pattern_tags:
                print(f"- {tag}")
        else:
            print("Tidak ada tag yang tersedia.")

    def clear_memory():
        """Hapus semua data dari memori."""
        global patterns, pattern_tags, embeddings, RESPONSES
        patterns = []
        pattern_tags = []
        embeddings = []
        RESPONSES = {}
        print("Semua data dalam memori telah dihapus.")
    
    #admin login system

    def admin_login():
        """Sistem login admin sederhana."""
        password = "admin123"  # Ganti dengan password yang lebih aman
        attempts = 3 

        while attempts > 0:
            user_input = input("Masukkan password admin: ")
            if user_input == password:
                print("Login berhasil! Selamat datang, Admin.")
                return True
            else:
                attempts -= 1
                print(f"Password salah. Sisa percobaan: {attempts}")
        print("Akses ditolak. Terlalu banyak percobaan salah.")
        return False
    # Cek apakah admin login
    if admin_login(): 
        while True:
            print("\nMenu Admin:")
            print("1. Hapus tag")
            print("2. Tampilkan semua tag")
            print("3. Hapus semua data memori")
            print("4. Keluar")
            choice = input("Pilih opsi (1-4): ")

            if choice == "1":
                tag_to_delete = input("Masukkan tag yang ingin dihapus: ").strip()
                delete_tag(tag_to_delete)
            elif choice == "2":
                list_tags()
            elif choice == "3":
                clear_memory()
            elif choice == "4":
                print("Keluar dari menu admin.")
                break
            else:
                print("Pilihan tidak valid. Silakan coba lagi.")