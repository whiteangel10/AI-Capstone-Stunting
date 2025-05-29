from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd

router = APIRouter()

model = joblib.load("models/model_mpasi.pkl")
scaler = joblib.load("models/scaler.pkl")
le_tekstur = joblib.load("models/le_tekstur.pkl")
le_kelas = joblib.load("models/le_kelas.pkl")

df_makanan = pd.read_excel("data/dataset_mpasi.xlsx")
df_makanan["kategori_umur"] = df_makanan["usia_min_bulan"].astype(str) + "-" + df_makanan["usia_max_bulan"].astype(str)

class FiturMakanan(BaseModel):
    kalori_kkal: float
    protein_gr: float
    lemak_gr: float
    tekstur: str
    jumlah_bahan: int

@router.post("/predict")
def predict(fitur: FiturMakanan):
    fitur_dict = fitur.dict()
    x = [
        fitur_dict['kalori_kkal'],
        fitur_dict['protein_gr'],
        fitur_dict['lemak_gr'],
        le_tekstur.transform([fitur_dict['tekstur']])[0],
        fitur_dict['jumlah_bahan']
    ]
    x_scaled = scaler.transform([x])
    pred = model.predict(x_scaled)
    kategori = le_kelas.inverse_transform(pred)[0].replace(" bulan", "").strip()

    df_filtered = df_makanan[df_makanan["kategori_umur"] == kategori]
    sampel = df_filtered.sample(n=min(5, len(df_filtered)))
    hasil = [{"nama": row["nama_makanan"], "resep": row["resep"]} for _, row in sampel.iterrows()]

    return {
        "input": fitur_dict,
        "prediksi_kategori_umur": kategori,
        "rekomendasi": hasil
    }
