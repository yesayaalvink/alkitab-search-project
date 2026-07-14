import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Tes AI Sederhana", layout="wide")

st.title("Aplikasi Tes AI Sederhana")
st.write("Aplikasi ini menggunakan model sangat ringan sebesar 80 megabyte untuk menguji apakah koneksi server Streamlit dan Hugging Face berjalan dengan lancar.")

# Memuat model uji coba ringan dari Hugging Face Hub
@st.cache_resource(show_spinner="Sedang memuat model uji coba ringan...")
def muat_model_uji():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

try:
    # Memanggil model
    model = muat_model_uji()
    st.success("Sukses! Koneksi berhasil dan model ringan telah dimuat ke dalam memori!")
    
    # Data uji coba statis tanpa perlu mengunduh berkas luar
    data_statis = [
        "Saya suka makan nasi goreng hangat",
        "Kucing lucu sedang tidur di atas meja",
        "Hari ini cuaca sangat cerah dan berawan"
    ]
    
    st.write("Daftar Kalimat di dalam Database Sementara:")
    for kalimat in data_statis:
        st.write(f"- {kalimat}")
        
    # Input dari pengguna
    input_user = st.text_input("Ketik kalimat acak untuk diuji kesamaan maknanya dengan daftar di atas:")
    
    if st.button("Uji Kesamaan Makna"):
        if input_user:
            # Mengonversi teks menjadi vektor
            vektor_input = model.encode(input_user, convert_to_tensor=True)
            vektor_data = model.encode(data_statis, convert_to_tensor=True)
            
            # Menghitung skor kemiripan
            skor = util.cos_sim(vektor_input, vektor_data)[0]
            indeks_terbaik = skor.argmax().item()
            
            st.success(f"Kalimat yang paling cocok secara makna adalah: **{data_statis[indeks_terbaik]}**")
        else:
            st.warning("Silakan isi teks terlebih dahulu.")
            
except Exception as e:
    st.error(f"Terjadi kesalahan koneksi! Detail kesalahan: {e}")
