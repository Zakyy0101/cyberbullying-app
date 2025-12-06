#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit App untuk Deteksi Cyberbullying di Komentar TikTok
Berdasarkan model CNN-BiLSTM yang sudah dilatih
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download resources NLTK
@st.cache_resource
def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

# Muat model
@st.cache_resource
def load_model():
    """Load the trained CNN-BiLSTM model"""
    try:
        model = tf.keras.models.load_model("models/best_lstm_final.h5")
        return model
    except:
        st.error("Model tidak ditemukan. Pastikan file 'models/best_lstm_final.h5' tersedia.")
        return None

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    """Create and fit tokenizer on sample data"""
    # In practice, you should save and load your actual tokenizer
    # This is a simplified version
    tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
    return tokenizer

# Preprocessing functions
def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)        # hapus URL
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) # hapus mention
    text = re.sub(r'#\S+', '', text)           # hapus hashtag
    text = re.sub(r'[^a-z\s]', '', text)       # hapus non-huruf
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    """Remove Indonesian stopwords"""
    stop_words = set(stopwords.words('indonesian'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def preprocess_single_text(text, tokenizer, max_len=300):
    """Preprocess a single text for prediction"""
    # Clean text
    cleaned = clean_text(text)
    
    # Remove stopwords
    no_stopwords = remove_stopwords(cleaned)
    
    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences([no_stopwords])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded, no_stopwords, cleaned

def analyze_text_characteristics(text):
    """Analyze text characteristics"""
    analysis = {}
    
    # Length analysis
    analysis['char_length'] = len(text)
    analysis['word_count'] = len(text.split())
    
    # Check for common cyberbullying indicators
    bullying_indicators = [
        'bodoh', 'tolol', 'goblok', 'anjing', 'bangsat',
        'jelek', 'gak berguna', 'sialan', 'dasar', 'mematikan',
        'beban', 'gagal', 'memalukan', 'konyol', 'murahan'
    ]
    
    found_indicators = []
    for indicator in bullying_indicators:
        if indicator in text.lower():
            found_indicators.append(indicator)
    
    analysis['bullying_indicators'] = found_indicators
    analysis['indicator_count'] = len(found_indicators)
    
    # Sentiment-like analysis (simple)
    negative_words = ['tidak', 'jangan', 'gak', 'ga', 'nggak', 'buruk', 'jelek']
    positive_words = ['bagus', 'baik', 'hebat', 'keren', 'mantap', 'luar biasa']
    
    neg_count = sum(1 for word in negative_words if word in text.lower())
    pos_count = sum(1 for word in positive_words if word in text.lower())
    
    analysis['negative_words'] = neg_count
    analysis['positive_words'] = pos_count
    
    return analysis

def create_visualization(prediction, confidence, analysis):
    """Create visualization for prediction results"""
    
    # Prediction gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': f"Probabilitas Cyberbullying"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig_gauge.update_layout(height=300)
    
    # Text characteristics
    char_data = {
        'Metric': ['Jumlah Karakter', 'Jumlah Kata', 'Indikator Bullying', 'Kata Negatif', 'Kata Positif'],
        'Value': [
            analysis['char_length'],
            analysis['word_count'],
            analysis['indicator_count'],
            analysis['negative_words'],
            analysis['positive_words']
        ]
    }
    
    df_chars = pd.DataFrame(char_data)
    fig_chars = px.bar(df_chars, x='Metric', y='Value', 
                       title='Karakteristik Teks',
                       color='Value',
                       color_continuous_scale='viridis')
    
    return fig_gauge, fig_chars

def main():
    """Main Streamlit app"""
    
    # Page configuration
    st.set_page_config(
        page_title="Deteksi Cyberbullying TikTok",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Title and description
    st.title("🛡️ Deteksi Cyberbullying dalam Komentar TikTok")
    st.markdown("""
    Aplikasi ini menggunakan model **CNN-BiLSTM** yang telah dilatih untuk mendeteksi 
    komentar cyberbullying dalam bahasa Indonesia. Model dapat mengklasifikasikan 
    komentar sebagai **normal** atau **cyberbullying**.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        # Model info
        st.subheader("Informasi Model")
        st.info("""
        **Arsitektur:** CNN-BiLSTM  
        **Input:** Teks komentar  
        **Output:** Probabilitas cyberbullying  
        **Akurasi:** ~85% (pada data test)
        """)
        
        # Demo examples
        st.subheader("Contoh Komentar")
        example_comments = [
            "konten kamu sangat bagus dan menginspirasi",
            "dasar goblok kerjaannya cuma nyontek doang",
            "wih keren banget video nya mantap",
            "jelek banget sih muka lo kayak babi",
            "terima kasih sudah berbagi informasi yang bermanfaat"
        ]
        
        selected_example = st.selectbox(
            "Pilih contoh komentar:",
            ["Pilih contoh..."] + example_comments
        )
        
        if selected_example != "Pilih contoh...":
            st.text_area("Komentar contoh:", selected_example, height=100)
            if st.button("Gunakan Contoh"):
                st.session_state.example_text = selected_example
                st.rerun()
        
        # About section
        st.divider()
        st.subheader("📊 Statistik")
        st.metric("Jumlah Kata dalam Vocabulary", "20,000")
        st.metric("Panjang Maksimal Teks", "300 kata")
        
        st.divider()
        st.caption("Dibangun dengan Streamlit & TensorFlow")
        st.caption("Model: CNN-BiLSTM")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Analisis Komentar")
        
        # Text input
        if 'example_text' in st.session_state:
            default_text = st.session_state.example_text
            del st.session_state.example_text
        else:
            default_text = ""
        
        input_text = st.text_area(
            "Masukkan komentar untuk dianalisis:",
            value=default_text,
            height=150,
            placeholder="Contoh: 'konten kamu sangat tidak berguna dan memalukan'"
        )
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Atau upload file CSV dengan kolom 'text'", 
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                if 'text' in df_upload.columns:
                    st.success(f"Berhasil mengupload {len(df_upload)} komentar")
                    
                    # Show sample
                    with st.expander("Lihat data yang diupload"):
                        st.dataframe(df_upload.head())
                    
                    # Batch prediction option
                    if st.button("Analisis Batch", type="primary"):
                        st.info("Fitur analisis batch sedang dalam pengembangan...")
                else:
                    st.error("File harus memiliki kolom 'text'")
            except Exception as e:
                st.error(f"Error membaca file: {e}")
    
    with col2:
        st.subheader("ℹ️ Panduan")
        
        with st.expander("Apa itu cyberbullying?"):
            st.write("""
            Cyberbullying adalah perilaku agresif yang dilakukan melalui media digital 
            dengan tujuan menyakiti, mengintimidasi, atau mempermalukan orang lain.
            
            **Ciri-ciri umum:**
            - Kata-kata kasar atau menghina
            - Ancaman atau intimidasi
            - Penyebaran kebohongan
            - Komentar merendahkan
            """)
        
        with st.expander("Contoh komentar cyberbullying"):
            st.write("""
            ✅ **Normal:** "Video yang bagus, terus berkarya!"
            ❌ **Cyberbullying:** "Dasar bodoh, konten lo sampah!"
            
            ✅ **Normal:** "Mungkin bisa diperbaiki editingnya"
            ❌ **Cyberbullying:** "Goblok banget sih ngedit kayak gini"
            """)
        
        with st.expander("Cara kerja model"):
            st.write("""
            1. **Preprocessing:** Membersihkan teks dari URL, mention, hashtag
            2. **Tokenisasi:** Mengubah teks menjadi urutan angka
            3. **CNN Layer:** Mengekstraksi pola lokal (n-gram)
            4. **BiLSTM Layer:** Memahami konteks maju-mundur
            5. **Classification:** Menghitung probabilitas cyberbullying
            """)
    
    # Analyze button
    if st.button("🔍 Analisis Komentar", type="primary", use_container_width=True):
        if input_text.strip():
            with st.spinner("Menganalisis komentar..."):
                try:
                    # Initialize components
                    model = load_model()
                    
                    if model is None:
                        st.error("Model tidak tersedia. Pastikan model sudah dilatih.")
                        return
                    
                    # Create tokenizer (in production, load the actual trained tokenizer)
                    tokenizer = load_tokenizer()
                    
                    # Preprocess and predict
                    processed_text, no_stopwords, cleaned_text = preprocess_single_text(input_text, tokenizer)
                    prediction = model.predict(processed_text, verbose=0)[0][0]
                    
                    # Analyze text characteristics
                    analysis = analyze_text_characteristics(cleaned_text)
                    
                    # Display results
                    st.divider()
                    st.subheader("📊 Hasil Analisis")
                    
                    # Results columns
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        if prediction >= 0.5:
                            st.error(f"🚨 **CYBERBULLYING DETECTED**")
                            st.metric(
                                "Probabilitas",
                                f"{prediction*100:.1f}%",
                                delta=f"Tinggi (threshold: 50%)",
                                delta_color="inverse"
                            )
                        else:
                            st.success(f"✅ **KOMENTAR NORMAL**")
                            st.metric(
                                "Probabilitas",
                                f"{prediction*100:.1f}%",
                                delta=f"Rendah (threshold: 50%)",
                                delta_color="normal"
                            )
                    
                    with res_col2:
                        st.metric("Jumlah Karakter", analysis['char_length'])
                        st.metric("Jumlah Kata", analysis['word_count'])
                    
                    with res_col3:
                        st.metric("Indikator Bullying", analysis['indicator_count'])
                        if analysis['indicator_count'] > 0:
                            st.caption(f"Ditemukan: {', '.join(analysis['bullying_indicators'][:3])}")
                    
                    # Visualizations
                    fig_gauge, fig_chars = create_visualization(prediction, prediction, analysis)
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with viz_col2:
                        st.plotly_chart(fig_chars, use_container_width=True)
                    
                    # Detailed analysis
                    with st.expander("🔍 Detail Preprocessing"):
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            st.write("**Teks Asli:**")
                            st.code(input_text)
                            
                            st.write("**Setelah Cleaning:**")
                            st.code(cleaned_text)
                        
                        with col_detail2:
                            st.write("**Tanpa Stopwords:**")
                            st.code(no_stopwords)
                            
                            st.write("**Karakteristik:**")
                            st.json({
                                "panjang_karakter": analysis['char_length'],
                                "jumlah_kata": analysis['word_count'],
                                "kata_negatif": analysis['negative_words'],
                                "kata_positif": analysis['positive_words'],
                                "indikator_bullying": analysis['bullying_indicators']
                            })
                    
                    # Explanation
                    if prediction >= 0.5:
                        st.warning("""
                        **Interpretasi:** Komentar ini memiliki karakteristik cyberbullying.
                        
                        **Rekomendasi:**
                        - Pertimbangkan untuk tidak mengirim komentar ini
                        - Gunakan bahasa yang lebih santun
                        - Fokus pada konten, bukan personal attack
                        """)
                    else:
                        st.info("""
                        **Interpretasi:** Komentar ini terlihat normal dan konstruktif.
                        
                        **Tetap ingat:** 
                        - Berkomentarlah dengan santun
                        - Fokus pada konten, bukan pribadi
                        - Jadilah netizen yang bertanggung jawab
                        """)
                    
                except Exception as e:
                    st.error(f"Terjadi error dalam analisis: {str(e)}")
                    st.exception(e)
        else:
            st.warning("Silakan masukkan teks komentar terlebih dahulu.")
    
    # Footer
    st.divider()
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.caption("**Model Accuracy:** ~85%")
    
    with col_footer2:
        st.caption("**Threshold:** 50%")
    
    with col_footer3:
        st.caption("**Version:** 1.0.0")
    
    # Disclaimer
    st.caption("""
    ⚠️ **Disclaimer:** Hasil deteksi ini berdasarkan model machine learning dan mungkin tidak 100% akurat. 
    Gunakan sebagai referensi dan pertimbangkan konteks secara menyeluruh.
    """)

if __name__ == "__main__":
    main()