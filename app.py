import streamlit as st
import pandas as pd
import re
import os
import glob
import kagglehub
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Pencarian Alkitab Semantik", layout="wide")

# Mengambil dan merapikan data secara otomatis
@st.cache_data(show_spinner="Sedang mengunduh dan merapikan data Alkitab...")
def siapkan_data():
    try:
        path_kaggle = kagglehub.dataset_download("williammulianto/indonesia-bible-tb")
        df_tb = pd.read_csv(os.path.join(path_kaggle, "tb.csv"))
        df_vmd = pd.read_csv(os.path.join(path_kaggle, "vmd.csv"))

        if not os.path.exists("ayt"):
            os.system("git clone https://github.com/sabdacode/ayt.git")
        file_csv_ayt = glob.glob("ayt/**/*.csv", recursive=True)[0]
        df_ayt = pd.read_csv(file_csv_ayt)

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

# Memuat model langsung dari Hugging Face Hub Anda
@st.cache_resource(show_spinner="Sedang memuat otak kecerdasan buatan dari Hugging Face...")
def muat_model():
    id_model = "YesayaAlvinK/indobert-bible-search"
    try:
        model = SentenceTransformer(id_model)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari Hugging Face Hub! Detail kesalahan: {e}")
        raise e

# Mengubah seluruh ayat menjadi vektor matematika
@st.cache_resource(show_spinner="Sedang memproses seluruh ayat menjadi vektor matematika (hanya dilakukan sekali)...")
def proses_vektor_ayat(_model, daftar_ayat):
    try:
        return _model.encode(daftar_ayat, show_progress_bar=True, convert_to_tensor=True)
    except Exception as e:
        st.error(f"Gagal memproses vektor ayat! Detail kesalahan: {e}")
        raise e

# Menjalankan proses inisialisasi
df_alkitab = siapkan_data()
model_ai = muat_model()
daftar_teks_tb = df_alkitab['teks_tb'].tolist()
vektor_seluruh_ayat = proses_vektor_ayat(model_ai, daftar_teks_tb)

# Membangun Antarmuka Pengguna
st.title("Mesin Pencari Alkitab Berbasis Makna")
st.write("Aplikasi ini menggunakan model IndoBERT yang telah dilatih mandiri oleh Anda menggunakan metode Multiple Negatives Ranking Loss untuk menemukan ayat berdasarkan makna konteks secara berdampingan.")

pertanyaan = st.text_input("Ketik pencarian Anda di sini (contoh: saudara Yusuf melemparkan Yusuf ke dalam sumur lalu menjualnya):")

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
