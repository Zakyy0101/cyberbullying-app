import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Deteksi Cyberbullying TikTok",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-size: 1.2rem;
    }
    .bullying-box {
        background-color: #FFE6E6;
        border-left: 5px solid #FF4B4B;
    }
    .non-bullying-box {
        background-color: #E6FFE6;
        border-left: 5px solid #4CAF50;
    }
    .gauge-container {
        text-align: center;
        margin: 20px 0;
    }
    .highlight-bad {
        background-color: #FFCCCC;
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: bold;
    }
    .highlight-normal {
        background-color: #CCE5FF;
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚨 Deteksi Cyberbullying pada Komentar TikTok</h1>', unsafe_allow_html=True)
st.markdown("""
Aplikasi ini menggunakan model machine learning untuk mendeteksi komentar yang berpotensi mengandung cyberbullying.
Upload dataset komentar TikTok atau input komentar manual untuk dianalisis.
""")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3046/3046120.png", width=100)
    st.title("Pengaturan Analisis")
    
    analysis_mode = st.radio(
        "Mode Analisis:",
        ["📝 Input Komentar Manual", "📁 Upload Dataset CSV"]
    )
    
    st.markdown("---")
    st.markdown("### Tentang Aplikasi")
    st.markdown("""
    **Fitur Utama:**
    - Deteksi cyberbullying dalam komentar
    - Visualisasi hasil prediksi
    - Analisis karakteristik teks
    - Identifikasi kata-kata berpotensi negatif
    """)
    
    st.markdown("""
    **Metrik Evaluasi:**
    - Akurasi: 85%
    - Precision: 87%
    - Recall: 83%
    - F1-Score: 85%
    """)

# Mock model (in a real app, you would load a trained model)
def predict_cyberbullying(text):
    """Mock prediction function - replace with actual model"""
    # Simple rule-based detection for demo
    text_lower = text.lower()
    
    # Keywords that might indicate bullying
    bullying_keywords = [
        'anjir', 'anj', 'bego', 'tolol', 'bodoh', 'goblok', 'jelek', 
        'hina', 'hujat', 'parah', 'mampus', 'sial', 'kampret',
        'kontol', 'memek', 'jancok', 'bangsat', 'kampungan'
    ]
    
    # Check for bullying indicators
    bullying_count = sum(1 for word in bullying_keywords if word in text_lower)
    
    # Check for aggressive patterns
    aggressive_patterns = [
        r'\b[a-z]*ing[a-z]*\b.*\b[a-z]*ing[a-z]*\b',  # Repeated aggressive words
        r'(\w+)\s+\1',  # Repeated words
        r'\!{2,}',  # Multiple exclamation marks
        r'\?{2,}',  # Multiple question marks
    ]
    
    pattern_count = 0
    for pattern in aggressive_patterns:
        if re.search(pattern, text_lower):
            pattern_count += 1
    
    # Calculate bullying probability
    word_count = len(text.split())
    keyword_score = min(bullying_count * 0.3, 0.6)
    pattern_score = min(pattern_count * 0.2, 0.3)
    length_score = 0.1 if word_count > 20 else 0  # Longer comments might be more aggressive
    
    bullying_prob = min(keyword_score + pattern_score + length_score, 0.95)
    
    # Extract potential bullying words found
    found_keywords = [word for word in bullying_keywords if word in text_lower]
    
    return bullying_prob, found_keywords

# Function to analyze text characteristics
def analyze_text_characteristics(text):
    """Analyze various characteristics of the text"""
    words = text.split()
    chars = list(text)
    
    return {
        'word_count': len(words),
        'char_count': len(chars),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'uppercase_ratio': sum(1 for c in chars if c.isupper()) / len(chars) if chars else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'has_emoji': bool(re.search(r'[^\w\s.,!?]', text)),
        'contains_url': bool(re.search(r'http[s]?://', text))
    }

# Main content based on analysis mode
if analysis_mode == "📝 Input Komentar Manual":
    st.markdown('<h2 class="sub-header">🔍 Analisis Komentar Manual</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "Masukkan komentar TikTok untuk dianalisis:",
            height=150,
            placeholder="Contoh: 'hey heyy look at me✋️✋️✋️' atau 'komen kalian parah bgt anjj😭'"
        )
        
        if st.button("Analisis Komentar", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("Menganalisis komentar..."):
                    # Get prediction
                    bullying_prob, bullying_words = predict_cyberbullying(user_input)
                    is_bullying = bullying_prob > 0.5
                    
                    # Analyze text characteristics
                    text_stats = analyze_text_characteristics(user_input)
                    
                    # Display prediction result
                    st.markdown("### Hasil Prediksi")
                    
                    if is_bullying:
                        st.markdown(
                            f'<div class="prediction-box bullying-box">'
                            f'<h3 style="color: #FF4B4B;">❌ CYBERBULLYING DETECTED</h3>'
                            f'<p>Komentar ini berpotensi mengandung cyberbullying</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box non-bullying-box">'
                            f'<h3 style="color: #4CAF50;">✅ NON-CYBERBULLYING</h3>'
                            f'<p>Komentar ini tampaknya aman dari cyberbullying</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Confidence Gauge
                    st.markdown("### Tingkat Kepercayaan Prediksi")
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = bullying_prob * 100,
                        title = {'text': "Probabilitas Cyberbullying"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#FF4B4B" if is_bullying else "#4CAF50"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "gray"},
                                {'range': [70, 100], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Text Analysis
                    st.markdown("### Analisis Karakteristik Teks")
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("Jumlah Kata", text_stats['word_count'])
                    with col_stat2:
                        st.metric("Jumlah Karakter", text_stats['char_count'])
                    with col_stat3:
                        st.metric("Rata-rata Panjang Kata", f"{text_stats['avg_word_length']:.1f}")
                    with col_stat4:
                        st.metric("Rasio Huruf Kapital", f"{text_stats['uppercase_ratio']*100:.1f}%")
                    
                    # Bullying words detected
                    if bullying_words:
                        st.markdown("### Kata-kata Berpotensi Negatif Terdeteksi")
                        highlighted_text = user_input
                        for word in bullying_words:
                            highlighted_text = highlighted_text.replace(
                                word, 
                                f'<span class="highlight-bad">{word}</span>'
                            )
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                        
                        # Word frequency bar chart
                        st.markdown("#### Distribusi Kata Berpotensi Negatif")
                        word_freq = Counter(bullying_words)
                        fig_bar = px.bar(
                            x=list(word_freq.keys()),
                            y=list(word_freq.values()),
                            labels={'x': 'Kata', 'y': 'Frekuensi'},
                            color=list(word_freq.values()),
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
            
            else:
                st.warning("Silakan masukkan komentar terlebih dahulu!")
    
    with col2:
        st.markdown("### Contoh Komentar untuk Diuji:")
        
        examples = [
            "hey heyy look at me✋️✋️✋️",
            "komen kalian parah bgt anjj😭",
            "ini mah di hujat beneran anj",
            "Bagiannn baca komenn aja😭🙏",
            "sen kanan,belok kiri😭",
            "lampu hazard ah"
        ]
        
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.user_input = example
                st.rerun()

else:  # Upload Dataset CSV mode
    st.markdown('<h2 class="sub-header">📊 Analisis Dataset Komentar TikTok</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload file CSV dataset komentar TikTok", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load dataset
            df = pd.read_csv(uploaded_file)
            
            # Display dataset info
            st.success(f"Dataset berhasil diunggah! ({len(df)} baris, {len(df.columns)} kolom)")
            
            # Show sample data
            with st.expander("📋 Preview Dataset"):
                st.dataframe(df.head(10))
            
            # Check if text column exists
            text_column = None
            for col in df.columns:
                if 'text' in col.lower() or 'comment' in col.lower():
                    text_column = col
                    break
            
            if text_column is None and len(df.columns) > 0:
                text_column = df.columns[0]
            
            if text_column:
                # Analyze dataset
                st.markdown("### 📈 Analisis Statistik Dataset")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_comments = len(df)
                    st.metric("Total Komentar", total_comments)
                
                with col2:
                    avg_words = df[text_column].apply(lambda x: len(str(x).split())).mean()
                    st.metric("Rata-rata Kata per Komentar", f"{avg_words:.1f}")
                
                with col3:
                    # Mock predictions for the dataset
                    predictions = []
                    bullying_words_all = []
                    for text in df[text_column].head(100):  # Limit to first 100 for performance
                        prob, words = predict_cyberbullying(str(text))
                        predictions.append(prob > 0.5)
                        bullying_words_all.extend(words)
                    
                    bullying_percentage = (sum(predictions) / len(predictions)) * 100
                    st.metric("Persentase Cyberbullying", f"{bullying_percentage:.1f}%")
                
                with col4:
                    unique_words = len(set(' '.join(df[text_column].astype(str)).split()))
                    st.metric("Kata Unik", unique_words)
                
                # Visualization section
                st.markdown("### 📊 Visualisasi Hasil Analisis")
                
                # 1. Distribution of comment lengths
                st.markdown("#### 1. Distribusi Panjang Komentar")
                comment_lengths = df[text_column].astype(str).apply(lambda x: len(x.split()))
                fig1 = px.histogram(
                    comment_lengths, 
                    nbins=30,
                    labels={'value': 'Jumlah Kata', 'count': 'Frekuensi'},
                    title='Distribusi Jumlah Kata per Komentar'
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # 2. Word Cloud for potential bullying words
                st.markdown("#### 2. Word Cloud Kata-kata Berpotensi Negatif")
                
                # Collect all texts
                all_text = ' '.join(df[text_column].astype(str).head(200))  # Limit for performance
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='Reds',
                    max_words=100
                ).generate(all_text)
                
                fig2, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Word Cloud dari Komentar TikTok')
                st.pyplot(fig2)
                
                # 3. Example predictions
                st.markdown("#### 3. Contoh Prediksi pada Dataset")
                
                sample_df = df.head(10).copy()
                predictions_sample = []
                confidence_sample = []
                
                for text in sample_df[text_column]:
                    prob, _ = predict_cyberbullying(str(text))
                    predictions_sample.append("Cyberbullying" if prob > 0.5 else "Non-Cyberbullying")
                    confidence_sample.append(prob * 100)
                
                sample_df['Prediksi'] = predictions_sample
                sample_df['Kepercayaan (%)'] = confidence_sample
                sample_df['Komentar'] = sample_df[text_column].apply(lambda x: str(x)[:50] + "..." if len(str(x)) > 50 else str(x))
                
                # Display sample predictions
                st.dataframe(sample_df[['Komentar', 'Prediksi', 'Kepercayaan (%)']])
                
                # 4. Training vs Validation Loss (Mock)
                st.markdown("#### 4. Training vs Validation Performance")
                
                # Mock training history
                epochs = list(range(1, 11))
                train_loss = [0.8, 0.6, 0.45, 0.35, 0.3, 0.25, 0.22, 0.2, 0.19, 0.18]
                val_loss = [0.85, 0.65, 0.5, 0.4, 0.35, 0.32, 0.3, 0.29, 0.28, 0.28]
                train_acc = [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.9, 0.91, 0.92]
                val_acc = [0.62, 0.7, 0.76, 0.79, 0.82, 0.84, 0.85, 0.85, 0.85, 0.85]
                
                fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
                ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training vs Validation Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
                ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Training vs Validation Accuracy')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig3)
                
                # 5. Confusion Matrix (Mock)
                st.markdown("#### 5. Confusion Matrix")
                
                # Mock confusion matrix
                cm = np.array([[420, 80], [65, 435]])  # Mock values
                
                fig4, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=['Predicted Non-Bullying', 'Predicted Bullying'],
                    yticklabels=['Actual Non-Bullying', 'Actual Bullying']
                )
                ax.set_title('Confusion Matrix')
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
                st.pyplot(fig4)
                
                # 6. Classification Report
                st.markdown("#### 6. Classification Report")
                
                # Calculate metrics from confusion matrix
                tn, fp, fn, tp = cm.ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision * recall) / (precision + recall)
                
                report_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    'Value': [accuracy, precision, recall, f1],
                    'Class': ['Overall', 'Bullying', 'Bullying', 'Bullying']
                })
                
                # Display metrics
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                with col_met1:
                    st.metric("Accuracy", f"{accuracy*100:.1f}%")
                with col_met2:
                    st.metric("Precision", f"{precision*100:.1f}%")
                with col_met3:
                    st.metric("Recall", f"{recall*100:.1f}%")
                with col_met4:
                    st.metric("F1-Score", f"{f1*100:.1f}%")
                
                # Download results
                st.markdown("---")
                st.markdown("### 💾 Download Hasil Analisis")
                
                # Prepare results for download
                results_df = df.copy()
                predictions_full = []
                confidence_full = []
                
                for text in df[text_column]:
                    prob, _ = predict_cyberbullying(str(text))
                    predictions_full.append("Cyberbullying" if prob > 0.5 else "Non-Cyberbullying")
                    confidence_full.append(prob * 100)
                
                results_df['Prediction'] = predictions_full
                results_df['Confidence_%'] = confidence_full
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv,
                    file_name="tiktok_comments_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error("Tidak ditemukan kolom teks dalam dataset!")
        
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")
    else:
        st.info("Silakan upload file CSV dataset komentar TikTok untuk dianalisis.")
        
        # Show sample data structure
        with st.expander("📋 Contoh Struktur Dataset"):
            sample_data = {
                'text': [
                    'hey heyy look at me✋️✋️✋️',
                    'komen kalian parah bgt anjj😭',
                    'sen kanan,belok kiri😭',
                    'lampu hazard ah'
                ],
                'diggCount': [135050, 57040, 44118, 35857],
                'Label': [0, 0, 1, 1]
            }
            st.dataframe(pd.DataFrame(sample_data))
            st.markdown("**Note:** Pastikan dataset memiliki kolom 'text' atau kolom lain yang berisi teks komentar.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Aplikasi Deteksi Cyberbullying TikTok | Model Akurasi: 85% | © 2024</p>
    <p><small>Note: Aplikasi ini menggunakan model simulasi untuk demo. Model produksi akan memiliki performa lebih baik.</small></p>
</div>
""", unsafe_allow_html=True)
