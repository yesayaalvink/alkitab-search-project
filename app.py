import os
import re
import glob
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(page_title="AI Bible Semantic Search", layout="wide")

st.title("📖 AI Bible Semantic Search")
st.subheader("Pencarian Ayat Alkitab Berbasis Makna Semantik")

# 1. Mengunduh dan menggabungkan data Alkitab secara otomatis
@st.cache_data
def muat_dan_gabung_data():
    import kagglehub
    
    # Mengunduh TB dan VMD dari Kaggle secara otomatis
    path_kaggle = kagglehub.dataset_download("williammulianto/indonesia-bible-tb")
    df_tb = pd.read_csv(os.path.join(path_kaggle, "tb.csv"))
    df_vmd = pd.read_csv(os.path.join(path_kaggle, "vmd.csv"))
    
    # Mengunduh AYT dari GitHub secara otomatis
    if not os.path.exists("ayt"):
        os.system("git clone https://github.com/sabdacode/ayt.git")
        
    file_csv_ayt = glob.glob("ayt/**/*.csv", recursive=True)
    df_ayt = pd.read_csv(file_csv_ayt[0])
    
    # Membersihkan tag kotor pada teks AYT
    def bersihkan_teks(teks):
        if pd.isna(teks):
            return ""
        return re.sub(r'<[^>]*>', '', str(teks)).strip()
        
    df_ayt['text_clean'] = df_ayt['text'].apply(bersihkan_teks)
    
    # Penyelarasan singkatan kitab Alkitab
    daftar_kitab_tb = df_tb['kitab'].unique()
    pemetaan_kitab = {i + 1: kitab for i, kitab in enumerate(daftar_kitab_tb)}
    df_ayt['kitab_standar'] = df_ayt['book'].map(pemetaan_kitab)
    
    # Memilah dan menyamakan nama kolom sebelum digabung
    tb_siap = df_tb[['kitab', 'pasal', 'ayat', 'firman']].rename(columns={'firman': 'teks_tb'})
    vmd_siap = df_vmd[['kitab', 'pasal', 'ayat', 'firman']].rename(columns={'firman': 'teks_vmd'})
    ayt_siap = df_ayt[['kitab_standar', 'chapter', 'verse', 'text_clean']].rename(
        columns={
            'kitab_standar': 'kitab',
            'chapter': 'pasal',
            'verse': 'ayat',
            'text_clean': 'teks_ayt'
        }
    )
    
    # Proses penggabungan final berdasarkan kunci komposit
    df_gabung_awal = pd.merge(tb_siap, vmd_siap, on=['kitab', 'pasal', 'ayat'], how='inner')
    df_final = pd.merge(df_gabung_awal, ayt_siap, on=['kitab', 'pasal', 'ayat'], how='inner')
    return df_final

with st.spinner("Sedang memuat database Alkitab dari server, mohon tunggu sebentar..."):
    df_alkitab_final = muat_dan_gabung_data()

# 2. Memuat model IndoBERT yang sudah dilatih dari Hugging Face Hub
# PENTING: Ganti tulisan username/model-name di bawah ini dengan akun dan repositori Hugging Face Anda
# Contoh: "william/indobert-bible-search"
NAMA_MODEL_HF = "YesayaAlvinK/indobert-bible-search"

@st.cache_resource
def muat_model_ai():
    return SentenceTransformer(NAMA_MODEL_HF)

@st.cache_resource
def proses_vektor_ayat(daftar_ayat):
    model = muat_model_ai()
    return model.encode(daftar_ayat, show_progress_bar=False)

try:
    with st.spinner("Sedang memuat model AI dari Hugging Face Hub..."):
        model_ai = muat_model_ai()
        
    # Menyiapkan daftar ayat untuk tiga versi terjemahan berbeda
    daftar_tb = df_alkitab_final['teks_tb'].tolist()
    daftar_vmd = df_alkitab_final['teks_vmd'].tolist()
    daftar_ayt = df_alkitab_final['teks_ayt'].tolist()
    
    with st.spinner("Sedang menyiapkan sistem pencarian cepat..."):
        vektor_seluruh_ayat = proses_vektor_ayat(daftar_tb)
        
    st.success("Sistem AI Pencarian Alkitab siap digunakan!")
    
    # 3. Bagian Antarmuka Tampilan Web
    st.write("Silakan masukkan konsep atau pernyataan makna yang ingin dicari di bawah ini:")
    kueri_pengguna = st.text_input("Input Pencarian Anda:", placeholder="Contoh: gembala menuntun ke air tenang")
    
    jumlah_tampil = st.slider("Jumlah Hasil yang Ingin Ditampilkan:", min_value=1, max_value=5, value=3)
    
    if st.button("Cari Ayat Semantik") and kueri_pengguna:
        with st.spinner("Sedang menghitung kedekatan makna ayat terdekat..."):
            vektor_pertanyaan = model_ai.encode(kueri_pengguna)
            skor_kemiripan = util.cos_sim(vektor_pertanyaan, vektor_seluruh_ayat)[0]
            
            # Mendapatkan urutan skor kemiripan tertinggi sesuai pilihan jumlah tampil
            indeks_terbaik_list = skor_kemiripan.argsort(descending=True)[:jumlah_tampil].tolist()
            
            for rank, indeks_terbaik in enumerate(indeks_terbaik_list):
                indeks_terbaik = int(indeks_terbaik)
                skor = skor_kemiripan[indeks_terbaik].item() * 100
                baris_data = df_alkitab_final.iloc[indeks_terbaik]
                
                st.markdown(f"### 📍 Hasil ke-{rank + 1} dengan Tingkat Kesamaan: {skor:.2f}%")
                st.markdown(f"**Lokasi Ayat:** {baris_data['kitab']} {baris_data['pasal']}:{baris_data['ayat']}")
                
                # Menampilkan tiga terjemahan secara sejajar dalam kolom
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info("**Terjemahan Baru**")
                    st.write(daftar_tb[indeks_terbaik])
                with col2:
                    st.warning("**Versi Mudah Dibaca**")
                    st.write(daftar_vmd[indeks_terbaik])
                with col3:
                    st.success("**Alkitab Yang Terbuka**")
                    st.write(daftar_ayt[indeks_terbaik])
                st.markdown("---")
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan Anda sudah mengganti variabel NAMA_MODEL_HF dengan tautan repositori Hugging Face Anda yang benar. Pesan kesalahan: {str(e)}")
