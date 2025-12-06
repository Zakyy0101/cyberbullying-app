import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

st.title("Cyberbullying Detection App 🚨")
st.write("Masukkan komentar TikTok kamu dan kami cek apakah ada unsur cyberbullying!")

# Load Model
model = load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 300

text = st.text_area("Masukkan komentar:")

if st.button("Prediksi"):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    pred = model.predict(padded)[0][0]

    if pred >= 0.5:
        st.error(f"Cyberbullying 😡 (score: {pred:.2f})")
    else:
        st.success(f"Non-Cyberbullying 😄 (score: {1-pred:.2f})")

