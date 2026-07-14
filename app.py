import streamlit as st
import pandas as pd
import re
import os
import gc
import torch
import numpy as np
import kagglehub
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Pencarian Alkitab Semantik", layout="wide")

# ==========================================
# CRITICAL FIX: Inject HF Token strictly
# If the token is missing, the app will stop immediately and tell you,
# rather than hanging silently due to rate limits.
# ==========================================
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
except KeyError:
    st.error("HF_TOKEN tidak ditemukan di Streamlit Secrets! Aplikasi tidak bisa mengunduh model.")
    st.stop()

# Limiting PyTorch threads to 1 to save server RAM
torch.set_num_threads(1)
gc.collect()

st.title("Mesin Pencari Alkitab Berbasis Makna")
st.write("Aplikasi ini menggunakan model IndoBERT yang telah dilatih mandiri oleh Anda menggunakan metode Multiple Negatives Ranking Loss untuk menemukan ayat berdasarkan makna konteks secara berdampingan.")

# Creating visual progress indicators on the main screen
st.subheader("Status Inisialisasi Sistem:")
status_data = st.empty()
status_model = st.empty()
status_vektor = st.empty()

# 1. Hugging Face Lock File Auto-Cleaner
hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(hf_cache_dir):
    for root, dirs, files in os.walk(hf_cache_dir):
        for file in files:
            if file.endswith(".lock"):
                try:
                    os.remove(os.path.join(root, file))
                except:
                    pass

# 2. Loading and preparing data automatically
@st.cache_data(show_spinner=None)
def siapkan_data():
    try:
        path_kaggle = kagglehub.dataset_download("williammulianto/indonesia-bible-tb")
        df_tb = pd.read_csv(os.path.join(path_kaggle, "tb.csv"))
        df_vmd = pd.read_csv(os.path.join(path_kaggle, "vmd.csv"))

        url_ayt_raw = "https://raw.githubusercontent.com/sabdacode/ayt/main/csv/ayt.csv"
        df_ayt = pd.read_csv(url_ayt_raw)

        def bersihkan_teks(teks):
            if pd.isna(teks): return ""
            return re.sub(r'<[^>]*>', '', str(teks)).strip()
        df_ayt['text_clean'] = df_ayt['text'].apply(bersihkan_teks)

        daftar_kitab_tb = df_tb['kitab'].unique()
        pemetaan_kitab = {i + 1: kitab for i, kitab in enumerate(daftar_kitab_tb)}
        df_ayt['kitab_standar'] = df_ayt['book'].map(pemetaan_kitab)

        tb_siap = df_tb[['kitab', 'pasal', 'ayat', 'firman']].rename(columns={'firman': 'teks_tb'})
        vmd_siap = df_vmd[['kitab', 'pasal', 'ayat', 'firman']].rename(columns={'firman': 'teks_vmd'})
        ayt_siap = df_ayt[['kitab_standar', 'chapter', 'verse', 'text_clean']].rename(
            columns={'kitab_standar': 'kitab', 'chapter': 'pasal', 'verse': 'ayat', 'text_clean': 'teks_ayt'}
        )

        df_gabung = pd.merge(tb_siap, vmd_siap, on=['kitab', 'pasal', 'ayat'], how='inner')
        df_final = pd.merge(df_gabung, ayt_siap, on=['kitab', 'pasal', 'ayat'], how='inner')
        return df_final
    except Exception as e:
        st.error(f"Gagal menyiapkan data Alkitab! Detail kesalahan: {e}")
        raise e

# 3. Loading the model directly from Hugging Face Hub with memory optimization
@st.cache_resource(show_spinner=None)
def muat_model():
    id_model = "YesayaAlvinK/indobert-bible-search"
    try:
        gc.collect()
        # CRITICAL: Pass token explicitly and use low_cpu_mem_usage to prevent OOM crash
        model = SentenceTransformer(
            id_model, 
            token=HF_TOKEN,
            device="cpu",
            model_kwargs={"low_cpu_mem_usage": True}
        )
        gc.collect()
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari Hugging Face Hub! Detail kesalahan: {e}")
        raise e

# 4. Encoding the entire corpus into mathematical vectors using BATCHING
@st.cache_resource(show_spinner=None)
def proses_vektor_ayat(_model, daftar_ayat):
    try:
        vektor_list = []
        ukuran_batch = 1500  # Smaller batch to strictly protect Streamlit's 1GB RAM limit
        total_ayat = len(daftar_ayat)
        
        progres_bar = st.progress(0.0)
        
        for i in range(0, total_ayat, ukuran_batch):
            batch = daftar_ayat[i : i + ukuran_batch]
            vektor_batch = _model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
            vektor_list.append(vektor_batch)
            
            gc.collect()
            
            persen = min(1.0, (i + ukuran_batch) / total_ayat)
            progres_bar.progress(persen)
            
        vektor_final_array = np.vstack(vektor_list)
        
        # Use from_numpy to share memory instead of duplicating it
        vektor_final_tensor = torch.from_numpy(vektor_final_array).float()
        
        # Clean up intermediate lists to free up RAM
        del vektor_final_array
        del vektor_list
        gc.collect()
        
        return vektor_final_tensor
    except Exception as e:
        st.error(f"Gagal memproses vektor ayat! Detail kesalahan: {e}")
        raise e

# Executing initialization with live progress monitoring
status_data.warning("🔄 Step 1: Downloading and aligning all Bible data...")
df_alkitab = siapkan_data()
status_data.success("✅ Step 1: All three versions of Bible data successfully prepared!")

status_model.warning("🔄 Step 2: Downloading IndoBERT model from Hugging Face... (Please wait 3-5 minutes)")
model_ai = muat_model()
status_model.success("✅ Step 2: IndoBERT model successfully loaded into memory!")

status_vektor.warning("🔄 Step 3: Processing thirty-one thousand verses into numeric vectors, batch process running...")
daftar_teks_tb = df_alkitab['teks_tb'].tolist()
vektor_seluruh_ayat = proses_vektor_ayat(model_ai, daftar_teks_tb)
status_vektor.success("✅ Step 3: All Bible vector coordinates successfully processed!")

st.write("---")

# Building user input interface
pertanyaan = st.text_input("Ketik pencarian Anda di sini, misalnya: saudara Yusuf melemparkan Yusuf ke dalam sumur lalu menjualnya")

if st.button("Cari Ayat"):
    if pertanyaan:
        vektor_pertanyaan = model_ai.encode(pertanyaan, convert_to_tensor=True)
        skor_kemiripan = util.cos_sim(vektor_pertanyaan, vektor_seluruh_ayat)[0]
        indeks_terbaik = skor_kemiripan.argmax().item()
        baris_hasil = df_alkitab.iloc[indeks_terbaik]

        st.success(f"Ayat paling relevan ditemukan di: **{baris_hasil['kitab']} {baris_hasil['pasal']}:{baris_hasil['ayat']}**")
        
        kolom1, kolom2, kolom3 = st.columns(3)
        with kolom1:
            st.subheader("Terjemahan Baru")
            st.info(baris_hasil['teks_tb'])
        with kolom2:
            st.subheader("Versi Mudah Dibaca")
            st.info(baris_hasil['teks_vmd'])
        with kolom3:
            st.subheader("Alkitab Yang Terbuka")
            st.info(baris_hasil['teks_ayt'])
    else:
        st.warning("Silakan ketik pertanyaan Anda terlebih dahulu.")
