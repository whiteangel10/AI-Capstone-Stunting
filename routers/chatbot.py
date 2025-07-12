from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import faiss
import re
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter()

# Load datasets
df_stunting = pd.read_csv("data/stunting_dataset.csv")
df_stunting['text'] = df_stunting['text'].astype(str)

df_mpasi = pd.read_excel("data/dataset_mpasi.xlsx")
df_mpasi['text'] = df_mpasi.apply(
    lambda row: f"{row['nama_makanan']} - {row['bahan']} - {row['tekstur']} - {row['cocok_untuk']} - {row['resep']}",
    axis=1
)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS index untuk stunting
stunting_embeddings = embedding_model.encode(df_stunting['text'].tolist(), convert_to_numpy=True)
stunting_embeddings = np.array(stunting_embeddings).astype('float32')
index_stunting = faiss.IndexFlatL2(stunting_embeddings.shape[1])
index_stunting.add(stunting_embeddings)

# Embedding MPASI
mpasi_embeddings = embedding_model.encode(df_mpasi['text'].tolist(), convert_to_numpy=True)
df_mpasi['embedding'] = list(mpasi_embeddings.astype('float32'))

# Pydantic Model
class MPASIQuestion(BaseModel):
    question: str
    usia_bulan: int = None
    user_id: int

# Ekstrak usia dari teks
def extract_usia_from_text(text):
    match = re.search(r"(\d+)\s*bulan", text.lower())
    if match:
        return int(match.group(1))
    if text.strip().isdigit():
        return int(text.strip())
    return None

# Batasi panjang context
def truncate_text(text, max_words=80):
    return ' '.join(text.split()[:max_words])

# Koneksi ke database
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="admin123$",
        database="laravel"
    )

def save_qa_to_db(pertanyaan, jawaban, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = "INSERT INTO chatbot (pertanyaan, jawaban, status) VALUES (%s, %s, %s)"
    cursor.execute(sql, (pertanyaan, jawaban, status))
    conn.commit()
    cursor.close()
    conn.close()

def save_unanswered_to_chat(pertanyaan, user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = "INSERT INTO chat (chat, chatby, user_id) VALUES (%s, 'user', %s)"
    cursor.execute(sql, (pertanyaan, user_id))
    conn.commit()
    cursor.close()
    conn.close()

# Deteksi relevansi pertanyaan
def is_relevant_to_stunting(text, threshold=0.3):
    keywords = [
        "stunting", "gizi", "pertumbuhan", "berat badan", "tumbuh kembang",
        "tinggi badan", "balita", "bayi", "mpasi", "makanan pendamping asi",
        "pencegahan", "nutrisi", "asi", "makanan sehat", "kekurangan gizi",
        "website stuntaid", "stuntaid"
    ]
    if any(kw in text.lower() for kw in keywords):
        return True
    topic_embedding = embedding_model.encode(["stunting, mpasi, gizi, pertumbuhan, asi, bayi"])
    input_embedding = embedding_model.encode([text])
    sim = cosine_similarity(input_embedding, topic_embedding)[0][0]
    return sim >= threshold

def generate_answer_stunting(question, user_id, threshold=2.5):
    if not is_relevant_to_stunting(question):
        return "Maaf, pertanyaan Anda tidak berkaitan dengan topik stunting atau MPASI.", "belum terjawab"

    query_vector = embedding_model.encode([question]).astype('float32')
    distances, indices = index_stunting.search(query_vector, 1)

    print(f"[DEBUG] Pertanyaan: {question}")
    print(f"[DEBUG] Jarak hasil: {distances[0][0]}")

    if distances[0][0] > threshold:
        jawaban = "Maaf, saya tidak menemukan informasi yang relevan untuk pertanyaan tersebut."
        status = "belum terjawab"
        save_qa_to_db(question, jawaban, status)
        save_unanswered_to_chat(question, user_id)
        return jawaban, status

    context = "\n".join(truncate_text(df_stunting.iloc[idx]['text']) for idx in indices[0])
    prompt = f"""
Kamu adalah asisten ahli stunting. Jawablah pertanyaan berikut secara singkat, jelas, dan akurat.

=== Informasi ===
{context}

=== Pertanyaan ===
{question}

=== Jawaban ===
"""
    res = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": "llama3", "prompt": prompt, "temperature": 0.3, "stream": False}
    )

    jawaban = res.json().get('response', 'Maaf, terjadi kesalahan saat menghasilkan jawaban.')
    status = "terjawab"
    save_qa_to_db(question, jawaban, status)
    return jawaban.strip(), status

def generate_answer_mpasi(question, usia_bulan, user_id=None, threshold=2.5):
    filtered_df = df_mpasi[
        (df_mpasi['usia_min_bulan'] <= usia_bulan) &
        (df_mpasi['usia_max_bulan'] >= usia_bulan)
    ]
    if filtered_df.empty:
        return f"Maaf, tidak ada rekomendasi MPASI untuk usia {usia_bulan} bulan.", "belum terjawab"

    context_texts = filtered_df['text'].tolist()
    embeddings_filtered = list(filtered_df['embedding'])
    index_filtered = faiss.IndexFlatL2(len(embeddings_filtered[0]))
    index_filtered.add(np.array(embeddings_filtered).astype('float32'))

    query_vector = embedding_model.encode([question]).astype('float32')
    distances, indices = index_filtered.search(query_vector, 1)

    print(f"[DEBUG] Pertanyaan MPASI: {question}")
    print(f"[DEBUG] Jarak hasil MPASI: {distances[0][0]}")

    if distances[0][0] > threshold:
        return f"Maaf, saya tidak menemukan rekomendasi MPASI yang sesuai untuk pertanyaan tersebut.", "belum terjawab"

    top_idx = indices[0][0]
    rekomendasi = filtered_df.iloc[top_idx]

    nama = rekomendasi['nama_makanan']
    bahan = rekomendasi['bahan'].replace('\n', ' ').replace('- ', '• ')
    cara = rekomendasi['resep'].replace('\n', '\n').replace('- ', '• ')

    response = f"""
**{nama}** (Usia {usia_bulan} bulan)

**Bahan:**
{bahan.strip()}

**Detail Bahan & Cara Membuat:**
{cara.strip()}
""".strip()

    status = "terjawab"
    save_qa_to_db(question, response, status)
    return response, status

# Fungsi utama penjawaban
def generate_answer_with_age(question, usia_bulan=None, user_id=None):
    detected_usia = extract_usia_from_text(question)
    if detected_usia is not None:
        return generate_answer_mpasi(question, detected_usia, user_id)

    if 'rekomendasi' in question.lower() and 'mpasi' in question.lower():
        if usia_bulan is None:
            return "Boleh tahu usia anak Anda (dalam bulan) agar saya bisa memberikan rekomendasi MPASI yang sesuai?", "terjawab"
        else:
            return generate_answer_mpasi(question, usia_bulan, user_id)

    return generate_answer_stunting(question, user_id)

@router.post("/ask")
def mpasi(q: MPASIQuestion):
    jawaban, status = generate_answer_with_age(q.question, q.usia_bulan, q.user_id)
    return {"jawaban": jawaban, "status": status}
