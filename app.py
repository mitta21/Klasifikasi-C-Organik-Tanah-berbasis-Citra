import streamlit as st
import cv2
import numpy as np
import pickle
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

st.set_page_config(
    page_title="SawitScan",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Responsif semua layar */
    .block-container {
        padding: 1.5rem 2rem;
        max-width: 1200px;
        margin: auto;
    }
    @media (max-width: 768px) {
        .block-container { padding: 1rem; }
    }

    /* Header */
    .header-box {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(27,67,50,0.3);
    }
    .header-box h1 {
        color: white;
        font-size: clamp(1.4rem, 3vw, 2.2rem);
        margin: 0 0 0.4rem 0;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .header-box p {
        color: #b7e4c7;
        margin: 0;
        font-size: clamp(0.8rem, 1.5vw, 1rem);
    }
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: #d8f3dc;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.8rem;
        margin-top: 0.8rem;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Card */
    .card {
        background: white;
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
        border: 1px solid #e9f5ee;
    }
    .card h3 {
        color: #1b4332;
        margin-top: 0;
        font-size: 1.1rem;
    }

    /* Step */
    .step-box {
        background: linear-gradient(90deg, #d8f3dc, #f0faf2);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        border-left: 4px solid #2d6a4f;
        font-size: 0.92rem;
        color: #1b4332;
    }

    /* Result card */
    .result-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border: 1px solid #e9f5ee;
        height: 100%;
    }
    .lapisan-title {
        font-weight: 700;
        font-size: 1rem;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .lapisan-depth {
        color: #74c69d;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }

    /* Badge kelas */
    .label-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        width: 90%;
    }
    .sangat-rendah { background:#ffebee; color:#c62828; border:2px solid #ef9a9a; }
    .rendah        { background:#fff3e0; color:#bf360c; border:2px solid #ffcc80; }
    .sedang        { background:#fffde7; color:#f57f17; border:2px solid #fff176; }
    .tinggi        { background:#e8f5e9; color:#1b5e20; border:2px solid #a5d6a7; }
    .sangat-tinggi { background:#e3f2fd; color:#0d47a1; border:2px solid #90caf9; }

    /* Ringkasan row */
    .summary-row {
        background: white;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        border: 1px solid #e9f5ee;
    }

    /* Keterangan kelas */
    .kelas-box {
        background: linear-gradient(135deg, #d8f3dc, #f0faf2);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.35rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        border: 1px solid #b7e4c7;
    }
    .kelas-box span { font-size: 1.2rem; }
    .kelas-info { flex: 1; }
    .kelas-name { font-weight: 700; color: #1b4332; font-size: 0.9rem; }
    .kelas-range { color: #52b788; font-size: 0.8rem; }

    /* Upload area */
    .upload-hint {
        background: #f0faf2;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        color: #52b788;
        font-size: 0.85rem;
        border: 2px dashed #b7e4c7;
        margin-top: 0.5rem;
    }

    /* Footer */
    .footer {
        background: linear-gradient(135deg, #1b4332, #2d6a4f);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 2rem;
        color: #b7e4c7;
        font-size: 0.85rem;
        line-height: 1.8;
    }
    .footer b { color: #d8f3dc; }

    /* Sembunyikan elemen bawaan streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with open('model_svm.pkl', 'rb') as f:
        return pickle.load(f)

model_data        = load_model()
hasil_per_lapisan = model_data['hasil_per_lapisan']
le                = model_data['label_encoder']
IMG_SIZE          = model_data['img_size']

# ============================================================
# FUNGSI EKSTRAKSI FITUR
# ============================================================
def extract_features(img_bgr):
    img_bgr = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_features = []
    for i in range(3):
        ch = img_hsv[:, :, i].flatten().astype(np.float32)
        hsv_features.append(np.mean(ch))
        hsv_features.append(np.std(ch))
        hsv_features.append(float(skew(ch)))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = (img_gray // 4).astype(np.uint8)
    glcm = graycomatrix(img_gray, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)
    glcm_features = []
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
        glcm_features.append(float(np.mean(graycoprops(glcm, prop))))
    return hsv_features + glcm_features

# ============================================================
# DATA KELAS
# ============================================================
info_kelas = {
    'sangat rendah': {'emoji':'🔴','css':'sangat-rendah','kadar':'< 1%',  'desc':'Sangat Rendah'},
    'rendah'       : {'emoji':'🟠','css':'rendah',       'kadar':'1 - 2%','desc':'Rendah'},
    'sedang'       : {'emoji':'🟡','css':'sedang',       'kadar':'2 - 3%','desc':'Sedang'},
    'tinggi'       : {'emoji':'🟢','css':'tinggi',       'kadar':'3 - 5%','desc':'Tinggi'},
    'sangat tinggi': {'emoji':'🔵','css':'sangat-tinggi','kadar':'> 5%',  'desc':'Sangat Tinggi'},
}
nama_lap  = {1:'Lapisan 1', 2:'Lapisan 2', 3:'Lapisan 3'}
kedalaman = {1:'0 – 20 cm', 2:'20 – 40 cm', 3:'40 – 60 cm'}

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-box">
    <h1>🌴 SawitScan</h1>
    <p>Prediksi Kadar C-Organik Tanah Sawit Berbasis Citra Digital</p>
    <div class="header-badge"> Powered by Support Vector Machine (SVM)</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LAYOUT UTAMA
# ============================================================
col_kiri, col_kanan = st.columns([1, 1.6], gap="large")

# ---- KOLOM KIRI ----
with col_kiri:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("###  Cara Penggunaan")
    for step in [
        "1️⃣  Siapkan foto tanah sawit tampak profil",
        "2️⃣  Pastikan 3 lapisan tanah terlihat jelas",
        "3️⃣  Upload foto di bawah ini",
        "4️⃣  Klik tombol <b>Analisis</b> dan lihat hasilnya"
    ]:
        st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("###  Upload Foto Tanah")
    uploaded_file = st.file_uploader(
        "Upload foto",
        type=['jpg','jpeg','png'],
        label_visibility='collapsed'
    )
    st.markdown("""
    <div class="upload-hint">
         Format: JPG, JPEG, PNG<br>
        Pastikan foto menampilkan 3 lapisan tanah secara vertikal
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=' Foto yang diupload', use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        tombol = st.button(" Analisis Kadar C-Organik",
                           type='primary', use_container_width=True)
    else:
        tombol = False

    # Keterangan kelas (selalu tampil di kiri bawah)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("###  Keterangan Kelas C-Organik")
    for kelas, info in info_kelas.items():
        st.markdown(f"""
        <div class="kelas-box">
            <span>{info['emoji']}</span>
            <div class="kelas-info">
                <div class="kelas-name">{info['desc']}</div>
                <div class="kelas-range">Kadar C-Organik: {info['kadar']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- KOLOM KANAN ----
with col_kanan:
    if uploaded_file and tombol:
        img_bgr   = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h         = img_bgr.shape[0]
        sepertiga = h // 3
        potongan  = {
            1: img_bgr[0:sepertiga, :],
            2: img_bgr[sepertiga:2*sepertiga, :],
            3: img_bgr[2*sepertiga:, :]
        }

        with st.spinner(' Sedang menganalisis citra tanah...'):
            hasil = {}
            for lap_num, pot in potongan.items():
                fitur  = extract_features(pot)
                scaled = hasil_per_lapisan[lap_num]['scaler'].transform([fitur])
                pred   = hasil_per_lapisan[lap_num]['model'].predict(scaled)
                label  = le.inverse_transform(pred)[0]
                hasil[lap_num] = {'label': label, 'pot': pot}

        st.success(" Analisis selesai!")

        # Hasil per lapisan
        st.markdown("###  Hasil Prediksi per Lapisan")
        c1, c2, c3 = st.columns(3, gap="small")

        for lap_num, col in zip([1,2,3], [c1,c2,c3]):
            label   = hasil[lap_num]['label']
            pot_rgb = cv2.cvtColor(
                cv2.resize(hasil[lap_num]['pot'], (120,120)),
                cv2.COLOR_BGR2RGB)
            info = info_kelas[label]
            with col:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.image(pot_rgb, use_column_width=True)
                st.markdown(f'<div class="lapisan-title">{nama_lap[lap_num]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="lapisan-depth"> {kedalaman[lap_num]}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="label-badge {info["css"]}">'
                    f'{info["emoji"]} {info["desc"].upper()}</span>',
                    unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Ringkasan tabel
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("###  Ringkasan Hasil")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        header_cols = st.columns([1.5, 1.5, 1, 1])
        header_cols[0].markdown("**Lapisan**")
        header_cols[1].markdown("**Kedalaman**")
        header_cols[2].markdown("**Kelas**")
        header_cols[3].markdown("**Kadar**")
        st.markdown("<hr style='margin:0.3rem 0; border-color:#d8f3dc'>", unsafe_allow_html=True)

        for lap_num in [1,2,3]:
            label = hasil[lap_num]['label']
            info  = info_kelas[label]
            row   = st.columns([1.5, 1.5, 1, 1])
            row[0].markdown(f"**{nama_lap[lap_num]}**")
            row[1].markdown(f"{kedalaman[lap_num]}")
            row[2].markdown(
                f'<span class="label-badge {info["css"]}" style="padding:0.2rem 0.6rem;font-size:0.8rem">'
                f'{info["emoji"]} {info["desc"]}</span>',
                unsafe_allow_html=True)
            row[3].markdown(f"**{info['kadar']}**")

        st.markdown('</div>', unsafe_allow_html=True)

    elif not uploaded_file:
        # Placeholder sebelum upload
        st.markdown('<div class="card" style="text-align:center;padding:3rem 1rem">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:4rem"></div>
        <h3 style="color:#2d6a4f">Selamat Datang di SawitScan</h3>
        <p style="color:#74c69d">
            Upload foto profil tanah sawit di sebelah kiri<br>
            untuk memulai analisis kadar C-Organik
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Info singkat
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("###  Tentang SawitScan")
        st.markdown("""
        <p style="color:#52b788; line-height:1.8">
        SawitScan adalah sistem prediksi kadar C-Organik tanah sawit 
        berbasis citra digital menggunakan algoritma 
        <b style="color:#1b4332">Support Vector Machine (SVM)</b>.<br><br>
        Sistem ini menganalisis <b style="color:#1b4332">3 lapisan tanah</b> 
        (0–20 cm, 20–40 cm, 40–60 cm) secara otomatis dari satu foto 
        profil tanah, menggunakan ekstraksi fitur 
        <b style="color:#1b4332">HSV Color Moment</b> dan 
        <b style="color:#1b4332">GLCM Tekstur</b>.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <b>SawitScan</b> — Sistem Prediksi Kadar C-Organik Tanah Sawit Berbasis Citra<br>
    <b>Institut Teknologi Sawit Indonesia</b><br>
    Fakultas Sains dan Teknologi &nbsp;|&nbsp; Jurusan Sistem dan Teknologi Informasi
</div>
""", unsafe_allow_html=True)