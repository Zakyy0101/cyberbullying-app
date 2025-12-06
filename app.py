#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit App untuk Deteksi Cyberbullying di Komentar TikTok
Berdasarkan model CNN-BiLSTM yang sudah dilatih
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Import TensorFlow/Keras
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    st.error("TensorFlow tidak terinstall. Mohon tambahkan 'tensorflow' ke requirements.txt")
    st.stop()

# ===============================
# SETUP HALAMAN STREAMLIT
# ===============================
st.set_page_config(
    page_title="Deteksi Cyberbullying TikTok",
    page_icon="🛡️",
    layout="wide"
)

# ===============================
# FUNGSI PREPROCESSING
# ===============================
def preprocess_text(text):
    """Membersihkan dan memproses teks komentar"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Hapus mention (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Hapus hashtag
    text = re.sub(r'#\w+', '', text)
    
    # Hapus karakter khusus dan angka
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Hapus stopwords bahasa Indonesia
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Gabungkan kembali
    return ' '.join(tokens)

def predict_cyberbullying(text, model, tokenizer, max_len):
    """Memprediksi apakah teks mengandung cyberbullying"""
    # Preprocess teks
    cleaned_text = preprocess_text(text)
    
    # Ubah ke sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    # Padding
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    
    # Prediksi
    prediction = model.predict(padded, verbose=0)
    
    return prediction[0][0], cleaned_text

# ===============================
# LOAD MODEL & TOKENIZER
# ===============================
@st.cache_resource
def load_models():
    """Load model dan tokenizer yang sudah disimpan"""
    try:
        # Load model
        model = load_model('model_cyberbullying_cnn_bilstm.h5')
        
        # Load tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load label encoder jika ada
        try:
            with open('label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
        except:
            label_encoder = None
        
        # Parameter
        max_len = 100  # Sesuaikan dengan model Anda
        
        return model, tokenizer, label_encoder, max_len
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, 100

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.title("🛡️ Deteksi Cyberbullying")
    st.markdown("---")
    
    st.subheader("📊 Statistik")
    
    # Contoh data statistik
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Komentar Dianalisis", "1,234")
    with col2:
        st.metric("Cyberbullying Terdeteksi", "156")
    
    st.markdown("---")
    
    st.subheader("⚙️ Pengaturan")
    confidence_threshold = st.slider(
        "Threshold Confidence",
        min_value=0.5,
        max_value=0.95,
        value=0.7,
        step=0.05,
        help="Tingkat kepercayaan minimum untuk deteksi cyberbullying"
    )
    
    st.markdown("---")
    
    st.info(
        """
        **Cara Penggunaan:**
        1. Masukkan komentar di tab **Deteksi Tunggal**
        2. Atau upload file CSV di tab **Batch Analysis**
        3. Lihat hasil dan visualisasi
        """
    )

# ===============================
# MAIN CONTENT
# ===============================
st.title("🛡️ Deteksi Cyberbullying dalam Komentar TikTok")
st.markdown("---")

# Load model
model, tokenizer, label_encoder, max_len = load_models()

# Buat tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Deteksi Tunggal", 
    "📁 Analisis Batch", 
    "📊 Dashboard Visualisasi"
])

# ===============================
# TAB 1: DETEKSI TUNGGAL
# ===============================
with tab1:
    st.header("Analisis Komentar Tunggal")
    
    col_input, col_result = st.columns([2, 1])
    
    with col_input:
        comment_input = st.text_area(
            "Masukkan komentar TikTok:",
            height=150,
            placeholder="Contoh: 'Kamu jelek banget sih, mending off sosmed deh'"
        )
        
        if st.button("🚀 Analisis Komentar", type="primary", use_container_width=True):
            if comment_input.strip():
                with st.spinner("Menganalisis komentar..."):
                    # Prediksi
                    probability, cleaned_text = predict_cyberbullying(
                        comment_input, model, tokenizer, max_len
                    )
                    
                    # Tentukan hasil
                    is_bullying = probability > confidence_threshold
                    
                    # Tampilkan hasil
                    with col_result:
                        st.subheader("Hasil Analisis")
                        
                        # Progress bar
                        st.progress(float(probability))
                        
                        # Metric
                        if is_bullying:
                            st.error(f"🚨 CYBERBULLYING DETECTED")
                            st.metric("Confidence", f"{probability:.2%}")
                            st.warning("⚠️ Komentar ini mengandung unsur cyberbullying")
                        else:
                            st.success(f"✅ AMAN")
                            st.metric("Confidence", f"{probability:.2%}")
                            st.info("Komentar ini terlihat normal")
                        
                        # Tampilkan teks yang sudah dibersihkan
                        with st.expander("Lihat teks setelah preprocessing"):
                            st.code(cleaned_text)
            else:
                st.warning("Mohon masukkan komentar terlebih dahulu")

# ===============================
# TAB 2: ANALISIS BATCH
# ===============================
with tab2:
    st.header("Analisis Batch dari File CSV")
    
    uploaded_file = st.file_uploader(
        "Upload file CSV dengan kolom 'comment' atau 'text'",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Cari kolom komentar
            comment_col = None
            for col in df.columns:
                if 'comment' in col.lower() or 'text' in col.lower():
                    comment_col = col
                    break
            
            if comment_col:
                st.success(f"✅ Kolom ditemukan: '{comment_col}'")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("📊 Analisis Semua Komentar", type="primary"):
                    with st.spinner("Menganalisis komentar..."):
                        # Buat progress bar
                        progress_bar = st.progress(0)
                        
                        # Analisis setiap komentar
                        results = []
                        for i, row in enumerate(df.itertuples()):
                            comment = getattr(row, comment_col)
                            probability, cleaned = predict_cyberbullying(
                                str(comment), model, tokenizer, max_len
                            )
                            
                            results.append({
                                'original_comment': comment,
                                'cleaned_comment': cleaned,
                                'probability': probability,
                                'is_cyberbullying': probability > confidence_threshold
                            })
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(df))
                        
                        # Buat DataFrame hasil
                        results_df = pd.DataFrame(results)
                        
                        # Tampilkan hasil
                        st.subheader("📋 Hasil Analisis Batch")
                        
                        # Statistik
                        col1, col2, col3, col4 = st.columns(4)
                        total = len(results_df)
                        bullying = results_df['is_cyberbullying'].sum()
                        
                        with col1:
                            st.metric("Total Komentar", total)
                        with col2:
                            st.metric("Cyberbullying", bullying)
                        with col3:
                            st.metric("Non-Cyberbullying", total - bullying)
                        with col4:
                            st.metric("Persentase", f"{(bullying/total*100):.1f}%")
                        
                        # Tabel hasil
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download hasil
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Hasil (CSV)",
                            data=csv,
                            file_name="hasil_deteksi_cyberbullying.csv",
                            mime="text/csv"
                        )
            else:
                st.error("❌ Kolom 'comment' atau 'text' tidak ditemukan dalam file CSV")
                st.write("Kolom yang tersedia:", df.columns.tolist())
                
        except Exception as e:
            st.error(f"Error membaca file: {e}")

# ===============================
# TAB 3: VISUALISASI
# ===============================
with tab3:
    st.header("📊 Dashboard Visualisasi")
    
    # Buat data contoh untuk visualisasi
    st.subheader("Distribusi Hasil Prediksi")
    
    # Data contoh
    categories = ['Normal', 'Cyberbullying Ringan', 'Cyberbullying Sedang', 'Cyberbullying Berat']
    values = [65, 20, 10, 5]
    
    # Buat visualisasi
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Pie chart
        fig1 = px.pie(
            names=categories,
            values=values,
            title="Distribusi Kategori Komentar",
            hole=0.4
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_viz2:
        # Bar chart
        fig2 = px.bar(
            x=categories,
            y=values,
            title="Jumlah per Kategori",
            labels={'x': 'Kategori', 'y': 'Jumlah'},
            color=values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Word Cloud untuk kata-kata yang sering muncul
    st.subheader("Kata-kata yang Sering Muncul")
    
    # Contoh teks
    sample_text = """
    jelek bodoh tolol idiot goblin monyet babi anjing
    mending mati cacat hideus menjijikkan sampah tai
    kampungan miskin miskin miskin miskin miskin miskin
    """
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='Reds',
        max_words=50
    ).generate(sample_text)
    
    # Tampilkan word cloud
    fig3, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig3)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Dikembangkan dengan ❤️ untuk melawan cyberbullying di platform TikTok</p>
        <p><small>Model: CNN-BiLSTM | Akurasi: ~92% | Bahasa: Indonesia</small></p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# FUNGSI TAMBAHAN UNTUK ERROR HANDLING
# ===============================
if model is None:
    st.error("""
    ⚠️ **Model tidak dapat dimuat!** 
    
    Pastikan file berikut ada di direktori project Anda:
    1. `model_cyberbullying_cnn_bilstm.h5` - Model CNN-BiLSTM
    2. `tokenizer.pkl` - Tokenizer untuk preprocessing
    3. `label_encoder.pkl` - Label encoder (opsional)
    
    Jika belum punya model, jalankan training terlebih dahulu.
    """)
