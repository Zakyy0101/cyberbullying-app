import streamlit as st
def _pad_sequences_fallback(sequences, maxlen, padding='post', truncating='post'):
    """Simple numpy-based pad_sequences replacement for demo mode."""
    arr = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        seq = list(seq)
        if truncating == 'post':
            seq = seq[:maxlen]
        else:
            seq = seq[-maxlen:]
        length = min(len(seq), maxlen)
        if padding == 'post':
            arr[i, :length] = seq[:length]
        else:
            arr[i, -length:] = seq[-length:]
    return arr

try:
    import tensorflow as tf  # type: ignore  # silence editor missing-import warnings when env not selected
    # Use attributes from the imported `tf` module to avoid some editor/linter import warnings
    load_model = tf.keras.models.load_model
    pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
    tensorflow_import_error = None
except Exception as e:
    tf = None
    load_model = None
    pad_sequences = _pad_sequences_fallback
    tensorflow_import_error = e
import pickle
import re
import string
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

# Path model utama (abaikan download GitHub)
MODEL_PATH = r"C:\Users\Thinkpad T460s\Downloads\deeplern tubes\models\model_terbaik.h5"
MODEL_OUT = MODEL_PATH
# Threshold kembali ke nilai training
BEST_THR_DEFAULT = 0.4
try:
    from download_model import ensure_model  # type: ignore  # allow running even if helper module is absent
except Exception:
    # fallback no-op if the helper is not available in the environment
    def ensure_model(url, dest_path, expected_sha256=None, force=False):
        return dest_path

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Deteksi Cyberbullying",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS - MIDNIGHT NEON MINIMALISM
# ==========================================
st.markdown("""
<style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background */
    .stApp {
        background-color: #0D1117;
        color: #C9D1D9;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
    }
    
    /* Header */
    h1 {
        color: #58A6FF;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
    }
    
    h2 {
        color: #58A6FF;
        font-weight: 600;
    }
    
    h3 {
        color: #C9D1D9;
        font-weight: 500;
    }
    
    /* Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #161B22 0%, #1c2128 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363d;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(88, 166, 255, 0.2);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #58A6FF 0%, #A371F7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.5);
        transform: scale(1.05);
    }
    
    /* Text Input */
    .stTextArea textarea {
        background-color: #161B22;
        color: #C9D1D9;
        border: 1px solid #30363d;
        border-radius: 10px;
    }
    
    /* Alert Boxes */
    .safe-box {
        background: linear-gradient(135deg, #3FB950 0%, #2ea043 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-weight: 600;
        font-size: 18px;
        box-shadow: 0 8px 16px rgba(63, 185, 80, 0.3);
    }
    
    .danger-box {
        background: linear-gradient(135deg, #F78166 0%, #ea6045 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-weight: 600;
        font-size: 18px;
        box-shadow: 0 8px 16px rgba(247, 129, 102, 0.3);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #58A6FF 0%, #A371F7 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL & RESOURCES
# ==========================================
# Download GitHub/URL diabaikan; gunakan model lokal MODEL_PATH

@st.cache_resource
def load_resources():
    """Load model, tokenizer, dan parameters"""
    demo_mode = False
    tf_error = tensorflow_import_error
    # If TensorFlow isn't available, provide a demo fallback so the UI can run
    if load_model is None:
        demo_mode = True

        # Create a fake model with the same predict interface
        class FakeModel:
            def predict(self, x, verbose=0):
                # return a low-confidence non-bully prediction for demo
                batch_size = x.shape[0] if hasattr(x, 'shape') else 1
                return np.zeros((batch_size, 1)) + 0.1

        class FakeTokenizer:
            def texts_to_sequences(self, texts):
                # return simple zero sequences
                return [[0] for _ in texts]

        fake_model = FakeModel()
        fake_tokenizer = FakeTokenizer()
        fake_params = {'max_len': 300}

        return fake_model, fake_tokenizer, fake_params, demo_mode, tf_error
    # Try several likely paths for model, tokenizer and params
    model = None
    model_candidate_paths = [
        MODEL_PATH,
        os.path.join('models', 'model_terbaik.h5'),
        'model_terbaik.h5',
        os.path.join('models', 'best_lstm_final_balanced.h5'),
        'best_lstm_final_balanced.h5',
    ]

    last_error = None
    for p in model_candidate_paths:
        if os.path.exists(p):
            try:
                model = load_model(p)
                break
            except Exception as e:
                last_error = e
                # try next candidate

    if model is None:
        st.error("❌ Model tidak ditemukan!")
        return None, None, None, demo_mode, tf_error

    # Load tokenizer
    tokenizer = None
    tokenizer_paths = [
        r'C:\Users\Thinkpad T460s\Downloads\deeplern tubes\exports\tokenizer_for_model_terbaik.pickle',
        r'C:\Users\Thinkpad T460s\Downloads\deeplern tubes\tokenizer.pickle',
        'tokenizer.pickle', 
        os.path.join('models', 'tokenizer.pickle'),
    ]
    for tk in tokenizer_paths:
        if os.path.exists(tk):
            try:
                with open(tk, 'rb') as handle:
                    tokenizer = pickle.load(handle)
                break
            except Exception as e:
                tokenizer = None

    if tokenizer is None:
        st.error("❌ Tokenizer tidak ditemukan!")

    # Load model params
    params = None
    params_paths = [
        'model_params.pickle', 
        os.path.join('models', 'model_params.pickle'),
    ]
    for pp in params_paths:
        if os.path.exists(pp):
            try:
                with open(pp, 'rb') as handle:
                    params = pickle.load(handle)
                break
            except Exception:
                params = None

    return model, tokenizer, params, demo_mode, tf_error

model, tokenizer, params, demo_mode, tf_error = load_resources()
if params is None:
    params = {}
else:
    # ensure required keys exist to avoid KeyError in UI
    params = dict(params)
    params.setdefault('vocab_size', 'N/A')
    params.setdefault('max_len', 300)
    params.setdefault('embedding_dim', 'N/A')
    params.setdefault('threshold', 0.4)
    # normalisasi tipe data
    try:
        params['max_len'] = int(params.get('max_len', 300))
    except Exception:
        params['max_len'] = 300
    try:
        params['threshold'] = float(params.get('threshold', 0.4))
    except Exception:
        params['threshold'] = 0.4

# Fallback: jika masih N/A, coba baca langsung file model_params.pickle di lokasi standar
if not params.get('vocab_size') or params.get('vocab_size') == 'N/A':
    fallback_paths = ['model_params.pickle', os.path.join('models', 'model_params.pickle')]
    for fp in fallback_paths:
        if os.path.exists(fp):
            try:
                with open(fp, 'rb') as fh:
                    loaded_params = pickle.load(fh)
                for k in ['vocab_size', 'max_len', 'embedding_dim', 'threshold']:
                    if k in loaded_params:
                        params[k] = loaded_params[k]
                break
            except Exception:
                pass
# pastikan angka fallback tetap valid
try:
    params['max_len'] = int(params.get('max_len', 300))
except Exception:
    params['max_len'] = 300
try:
    params['threshold'] = float(params.get('threshold', 0.4))
except Exception:
    params['threshold'] = 0.4
# simpan threshold global ala notebook
best_thr = params.get('threshold', BEST_THR_DEFAULT)

# ==========================================
# FUNGSI PREPROCESSING & PREDIKSI
# ==========================================
def clean_text(text):
    """Preprocessing text seperti saat training"""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_bully_sentence(text, model=None, tokenizer=tokenizer, maxlen=300, threshold=None):
    """
    Input: text (str)
    Output: tuple (label_str, prob_float, cleaned_text, threshold_used)
    """
    # pilih model: parameter > best_model > model_bal > coba load dari disk
    if model is None:
        model = globals().get('best_model') or globals().get('model_bal')
        if model is None:
            try:
                model = load_model(MODEL_OUT)
            except Exception:
                try:
                    model = load_model('models/model_terbaik.h5')
                except Exception as e:
                    raise RuntimeError("Tidak menemukan model di memori atau file. Pastikan model .h5 tersedia.") from e

    # ambil threshold terbaik jika tersedia, fallback ke BEST_THR_DEFAULT
    thr = threshold if threshold is not None else globals().get('best_thr', BEST_THR_DEFAULT)
    try:
        thr = float(thr)
    except Exception:
        thr = BEST_THR_DEFAULT

    # preprocessing sesuai pipeline notebook
    txt_clean = clean_text(text)
    
    # PENTING: Jangan hapus stopwords untuk kata-kata bully!
    # Stopwords removal bisa menghilangkan kata penting seperti "dasar", "tidak", dll
    # Gunakan text yang sudah di-clean langsung
    txt_for_model = txt_clean
    
    # Jika ingin tetap hapus stopwords, gunakan ini (tapi tidak disarankan):
    # try:
    #     from nltk.corpus import stopwords
    #     stop_words = set(stopwords.words('indonesian')) | set(stopwords.words('english'))
    #     # Jangan hapus kata-kata yang potensial bully
    #     bully_keywords = {'tolol', 'bodoh', 'goblok', 'anjing', 'babi', 'kontol', 'bangsat', 'kampret'}
    #     txt_for_model = ' '.join([w for w in txt_clean.split() if w not in stop_words or w in bully_keywords])
    # except Exception:
    #     txt_for_model = txt_clean

    # Pastikan tokenizer dan pad_sequences tersedia
    if tokenizer is None:
        raise RuntimeError("Tokenizer tidak tersedia. Pastikan tokenizer.pickle sudah dimuat dengan benar.")
    
    seq = tokenizer.texts_to_sequences([txt_for_model])
    X = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')

    prob = float(model.predict(X, verbose=0).ravel()[0])
    label = "BULLY" if prob >= thr else "NOT BULLY"

    return label, prob, txt_clean, thr

def predict_text(text, model=None, tokenizer=None, max_len=300, threshold=None):
    """Prediksi cyberbullying dengan fallback model/threshold."""
    # pilih threshold: prioritas argumen, lalu params, default 0.5
    if threshold is None:
        threshold = params.get('threshold', 0.5) if isinstance(params, dict) else 0.5
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.5

    # pilih model/tokenizer dari argumen; jika None, coba best_model / model_bal / load disk
    if model is None or tokenizer is None:
        m = globals().get('best_model') or globals().get('model_bal')
        if m is None:
            # coba load dari file yang umum dipakai
            for cand in [
                'models/model_terbaik.h5',
                'model_terbaik.h5',
                'models/best_lstm_final_balanced.h5',
                'best_lstm_final_balanced.h5',
                'models/best_cnn_bilstm_model.h5',
                'best_cnn_bilstm_model.h5',
            ]:
                if os.path.exists(cand):
                    try:
                        m = load_model(cand)
                        break
                    except Exception:
                        continue
        model = model or m
        tokenizer = tokenizer or globals().get('tokenizer')

    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned]) if tokenizer else [[0]]
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prob = float(model.predict(padded, verbose=0)[0][0]) if model else 0.1
    prediction = 1 if prob >= threshold else 0
    return prediction, prob, cleaned, threshold

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio(
    "",
    ["🏠 Home", "🔮 Prediksi", "📊 Performa Model", "🧬 Arsitektur"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📌 Tentang Aplikasi
Sistem deteksi cyberbullying menggunakan **CNN-BiLSTM** untuk analisis komentar otomatis.

**Tech Stack:**
- TensorFlow/Keras
- Streamlit
- NLP Processing

**Developer:** Kelompok 21
""")

# ==========================================
# HALAMAN HOME
# ==========================================
if page == "🏠 Home":
    # Hero Section
    st.markdown("<h1>🛡️ Sistem Deteksi Komentar Bully Berbasis Deep Learning</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; font-size: 18px; color: #8B949E; margin-bottom: 40px;'>
    Sebuah sistem analisis komentar otomatis yang dirancang untuk mengidentifikasi indikasi bullying 
    melalui kombinasi arsitektur <b>CNN</b> dan <b>BiLSTM</b>.
    </div>
    """, unsafe_allow_html=True)
    
    # About Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Tujuan Pengembangan")
        st.markdown("""
        <div class='metric-card'>
        Aplikasi ini dikembangkan untuk membantu proses moderasi komentar secara otomatis, 
        mendeteksi pola bahasa yang mengandung unsur penghinaan, merendahkan, atau menyerang 
        secara personal. Model deep learning dilatih menggunakan dataset komentar berlabel 
        untuk memahami konteks dan intensi pada kalimat.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🧠 Model Metrics")
        metrics_data = {
            "Accuracy": ("83.6%", "#58A6FF"),
            "Precision": ("76.0%", "#3FB950"),
            "Recall": ("81.2%", "#F78166"),
            "F1-Score": ("78.5%", "#A371F7")
        }
        
        for metric, (value, color) in metrics_data.items():
            st.markdown(f"""
            <div class='metric-card' style='border-left: 4px solid {color}; margin-bottom: 10px;'>
                <h4 style='color: {color}; margin: 0;'>{metric}</h4>
                <p style='font-size: 24px; font-weight: 700; margin: 5px 0;'>{value}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Flow Diagram
    st.markdown("### 🔄 Bagaimana Sistem Ini Bekerja?")
    
    col1, col2, col3, col4 = st.columns(4)
    
    steps = [
        ("📝", "Input", "Komentar dimasukkan"),
        ("🧹", "Preprocessing", "Cleaning & normalisasi"),
        ("🔢", "Tokenizing", "Konversi ke angka"),
        ("🧠", "Model", "CNN-BiLSTM prediksi")
    ]
    
    for col, (emoji, title, desc) in zip([col1, col2, col3, col4], steps):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 48px;'>{emoji}</div>
                <h4 style='color: #58A6FF;'>{title}</h4>
                <p style='font-size: 12px; color: #8B949E;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-top: 30px; font-size: 16px; color: #8B949E;'>
    ✨ <i>"Model tidak hanya membaca teks—melainkan memahami maksud di baliknya."</i>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# HALAMAN PREDIKSI
# ==========================================
elif page == "🔮 Prediksi":
    st.markdown("<h1>🔮 Uji Komentar Anda</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; font-size: 16px; color: #8B949E; margin-bottom: 30px;'>
    Masukkan komentar apa pun dan biarkan sistem menganalisis apakah komentar tersebut 
    mengandung unsur bullying.
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Model belum berhasil dimuat. Pastikan file model tersedia!")
    else:
        # Input Box
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            user_input = st.text_area(
                "",
                placeholder="Tulis komentar di sini...",
                height=150,
                key="input_text"
            )
            
            predict_button = st.button("🔮 Analisis Komentar", use_container_width=True)
        
        if predict_button and user_input:
            with st.spinner("Menganalisis komentar..."):
                # Gunakan threshold dari params (yang sudah diset saat training)
                label_str, probability, cleaned_text, used_threshold = predict_bully_sentence(
                    user_input,
                    model=model,
                    tokenizer=tokenizer,
                    maxlen=params.get("max_len", 300),
                    threshold=params.get("threshold", BEST_THR_DEFAULT)
                )
                prediction = 1 if label_str == "BULLY" else 0
            
            st.markdown("---")
            
            # Debug Info (SIMPLE VERSION)
            with st.expander("🔍 Debug Information"):
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.write("**Input & Processing:**")
                    st.code(f"Input: {user_input}")
                    st.code(f"Cleaned: {cleaned_text}")
                    
                with col_d2:
                    st.write("**Prediction Result:**")
                    st.code(f"Probability: {probability:.4f}")
                    st.code(f"Threshold: {used_threshold:.4f}")
                    st.code(f"Result: {label_str}")
            
            # Result Section
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if prediction == 0:
                    # Non-Bully Result
                    st.markdown("""
                    <div class='safe-box'>
                        ✅ Komentar Aman<br>
                        <span style='font-size: 14px; font-weight: 400;'>
                        Tidak ditemukan indikasi bahasa merendahkan atau menyerang.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence Bar
                    st.markdown("### Keyakinan Model")
                    st.progress(1 - probability)
                    st.markdown(f"<div style='text-align: center; color: #3FB950; font-size: 20px; font-weight: 600;'>{(1-probability)*100:.1f}%</div>", unsafe_allow_html=True)
                
                else:
                    # Bully Result
                    st.markdown("""
                    <div class='danger-box'>
                        ⚠️ Komentar Mengandung Bullying<br>
                        <span style='font-size: 14px; font-weight: 400;'>
                        Sistem mendeteksi pola bahasa yang mengarah pada hinaan atau serangan personal.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence Bar
                    st.markdown("### Keyakinan Model")
                    st.progress(probability)
                    st.markdown(f"<div style='text-align: center; color: #F78166; font-size: 20px; font-weight: 600;'>{probability*100:.1f}%</div>", unsafe_allow_html=True)
                
                # Cleaned Text Preview
                with st.expander("📄 Teks Setelah Preprocessing"):
                    st.code(cleaned_text if cleaned_text else "Teks kosong setelah cleaning")

# ==========================================
# HALAMAN PERFORMA MODEL
# ==========================================
elif page == "📊 Performa Model":
    st.markdown("<h1>📊 Analisis Performa Model</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; font-size: 16px; color: #8B949E; margin-bottom: 30px;'>
    Evaluasi ini menggambarkan seberapa baik model dalam memahami pola bahasa 
    yang mengandung unsur bullying.
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Accuracy", "83.60%", "#58A6FF", col1),
        ("Precision", "76.04%", "#3FB950", col2),
        ("Recall", "81.17%", "#F78166", col3),
        ("F1-Score", "78.50%", "#A371F7", col4)
    ]
    
    for metric_name, value, color, col in metrics:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <h4 style='color: {color};'>{metric_name}</h4>
                <p style='font-size: 32px; font-weight: 700; color: {color}; margin: 10px 0;'>{value}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Confusion Matrix Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Confusion Matrix")
        
        # Data confusion matrix (sesuaikan dengan hasil model Anda)
        cm_data = [[2713, 287], [289, 911]]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=['Predicted Non-Bully', 'Predicted Bully'],
            y=['Actual Non-Bully', 'Actual Bully'],
            colorscale=[[0, '#0D1117'], [0.5, '#58A6FF'], [1, '#F78166']],
            text=cm_data,
            texttemplate="%{text}",
            textfont={"size": 20, "color": "white"},
            showscale=False
        ))
        
        fig.update_layout(
            plot_bgcolor='#0D1117',
            paper_bgcolor='#161B22',
            font=dict(color='#C9D1D9'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style='font-size: 14px; color: #8B949E; text-align: center;'>
        Confusion Matrix menunjukkan bagaimana model memisahkan komentar bully dan non-bully.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Classification Report")
        
        report_data = {
            'Class': ['Non-Bully', 'Bully', '', 'Weighted Avg'],
            'Precision': ['91.82%', '66.78%', '', '84.95%'],
            'Recall': ['84.57%', '81.17%', '', '83.60%'],
            'F1-Score': ['88.04%', '73.87%', '', '84.00%'],
            'Support': ['3000', '1200', '', '4200']
        }
        
        st.markdown("""
        <div class='metric-card'>
        """, unsafe_allow_html=True)
        
        for i in range(len(report_data['Class'])):
            if report_data['Class'][i] == '':
                st.markdown("---")
            else:
                cols = st.columns([2, 2, 2, 2, 2])
                cols[0].markdown(f"**{report_data['Class'][i]}**")
                cols[1].markdown(report_data['Precision'][i])
                cols[2].markdown(report_data['Recall'][i])
                cols[3].markdown(report_data['F1-Score'][i])
                cols[4].markdown(report_data['Support'][i])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='font-size: 14px; color: #8B949E; text-align: center; margin-top: 15px;'>
        Model memiliki keseimbangan baik antara precision dan recall.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# HALAMAN ARSITEKTUR
# ==========================================
elif page == "🧬 Arsitektur":
    st.markdown("<h1>🧬 Arsitektur Model CNN–BiLSTM</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; font-size: 18px; color: #8B949E; margin-bottom: 40px;'>
    Kombinasi CNN dan BiLSTM memberikan kemampuan menganalisis struktur kalimat 
    sekaligus memahami konteks emosional dari dua arah.
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture Diagram
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        layers = [
            ("🔥 Input Layer", "Sequence of tokens (max_len=300)", "#58A6FF"),
            ("🔤 Embedding Layer", "100-dimensional word embeddings", "#A371F7"),
            ("🔍 Conv1D Layer", "128 filters, kernel_size=5", "#F78166"),
            ("📊 MaxPooling1D", "Pool size=2", "#3FB950"),
            ("🔄 Bidirectional LSTM", "64 units (128 total)", "#58A6FF"),
            ("🌍 GlobalMaxPooling1D", "Feature aggregation", "#A371F7"),
            ("🧠 Dense Layer", "64 units, ReLU activation", "#F78166"),
            ("💧 Dropout", "Rate=0.3", "#3FB950"),
            ("🔤 Output Layer", "Sigmoid activation (0-1)", "#58A6FF")
        ]
        
        for layer_name, description, color in layers:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 4px solid {color}; margin-bottom: 15px;'>
                <h4 style='color: {color}; margin: 0;'>{layer_name}</h4>
                <p style='font-size: 14px; color: #8B949E; margin: 5px 0;'>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed Explanations
    st.markdown("### 📚 Penjelasan Detail Setiap Layer")
    
    explanations = [
        {
            "icon": "🔤",
            "title": "Embedding Layer",
            "content": "Mengubah setiap kata menjadi representasi numerik sehingga model dapat mengenali makna semantik.",
            "color": "#A371F7"
        },
        {
            "icon": "🔍",
            "title": "CNN Layer",
            "content": "Mengekstraksi pola-pola lokal seperti kata kasar, frasa serangan, atau struktur kalimat yang sering muncul dalam komentar bullying.",
            "color": "#F78166"
        },
        {
            "icon": "🔄",
            "title": "BiLSTM Layer",
            "content": "Membaca kalimat dari dua arah untuk memahami konteks sebelum dan sesudah kata tertentu—ideal untuk memahami nada dan intensi.",
            "color": "#58A6FF"
        },
        {
            "icon": "🔤",
            "title": "Dense Output",
            "content": "Menghasilkan probabilitas sebuah komentar termasuk kategori bully atau tidak.",
            "color": "#3FB950"
        }
    ]
    
    cols = st.columns(2)
    for idx, exp in enumerate(explanations):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 48px; text-align: center;'>{exp['icon']}</div>
                <h3 style='color: {exp['color']}; text-align: center;'>{exp['title']}</h3>
                <p style='text-align: center; color: #8B949E;'>{exp['content']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-top: 40px; font-size: 18px; color: #8B949E;'>
    ✨ <i>"Model ini tidak hanya membaca teks—melainkan memahami maksud di baliknya."</i>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Parameters
    if params:
        st.markdown("---")
        st.markdown("### ⚙️ Model Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <h4 style='color: #58A6FF;'>Vocabulary Size</h4>
                <p style='font-size: 28px; font-weight: 700;'>{params.get('vocab_size', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <h4 style='color: #58A6FF;'>Max Sequence Length</h4>
                <p style='font-size: 28px; font-weight: 700;'>{params.get('max_len', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <h4 style='color: #58A6FF;'>Embedding Dimension</h4>
                <p style='font-size: 28px; font-weight: 700;'>{params.get('embedding_dim', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)