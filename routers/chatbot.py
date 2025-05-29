from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import requests

router = APIRouter()

df = pd.read_csv("data/stunting_dataset.csv")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df['text'].tolist())
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

class Question(BaseModel):
    question: str

def retrieve_context(query, top_k=3):
    query_vector = embedding_model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    return "\n".join(df.iloc[idx]['text'] for idx in indices[0])

def generate_answer(question):
    context = retrieve_context(question)
    prompt = f"""
    Anda adalah asisten ahli tentang stunting.
    Jawablah pertanyaan berikut dengan ringkas dan jelas berdasarkan data berikut:

    {context}

    Pertanyaan: {question}
    Jawaban:
    """
    res = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": "llama3", "prompt": prompt, "temperature": 0.3, "stream": False}
    )
    return res.json()['response']

@router.post("/ask")
def ask(q: Question):
    jawaban = generate_answer(q.question)
    return {"jawaban": jawaban}
