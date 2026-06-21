import os
import re
import json
import base64
import pickle

import numpy as np
import pandas as pd
import emoji
import streamlit as st
import plotly.express as px

script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, '..', 'code', 'xgboost_judol_pipeline.pkl')

st.set_page_config(page_title="Sistem Analisis Sentimen Judi Online", layout="wide")


def get_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def load_background():
    path = os.path.join(script_dir, 'judol.png')
    if not os.path.exists(path):
        return
    data = get_base64(path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{data}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_css():
    path = os.path.join(script_dir, 'style.css')
    if os.path.exists(path):
        with open(path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def load_json(name):
    path = os.path.join(script_dir, name)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def clean_text(text):
    text = re.sub(r'@\w+', '', str(text))
    text = re.sub('<USERNAME>', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text.lower().strip())
    text = re.sub('#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess(text, kamus_norm, kamus_stop, kamus_stem):
    tokens = clean_text(text).split()
    if kamus_norm:
        tokens = [kamus_norm.get(w.lower(), w) for w in tokens]
    if kamus_stop:
        tokens = [w for w in tokens if w.lower() not in kamus_stop]
    if kamus_stem:
        tokens = [kamus_stem.get(w.lower(), w) for w in tokens]
    return ' '.join(tokens)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_kamus():
    norm = load_json('kamus_normalisasi.json')
    stem = load_json('kamus_stemming.json')
    stop = load_json('kamus_stopword.json')
    stop = set(stop) if stop else None
    return norm, stop, stem


def predict(texts, pipeline, kamus):
    norm, stop, stem = kamus
    siap = [preprocess(t, norm, stop, stem) for t in texts]
    X = pipeline['tfidf'].transform(siap)
    encoded = pipeline['model'].predict(X)
    proba = pipeline['model'].predict_proba(X)
    le = pipeline['label_encoder']
    labels = le.inverse_transform(encoded)
    classes = le.inverse_transform(pipeline['model'].classes_)
    return siap, labels, proba, classes


load_background()
load_css()

st.title("Sistem Analisis Sentimen Judi Online")
st.caption("Klasifikasi sentimen teks pengguna X menggunakan model XGBoost terlatih.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan di {MODEL_PATH}. Latih model pada notebook dan simpan sebagai xgboost_judol_pipeline.pkl.")
    st.stop()

pipeline = load_model()
kamus = load_kamus()
COLORS = {'negatif': '#F44336', 'netral': '#2196F3', 'positif': '#4CAF50'}

tab_single, tab_batch, tab_model = st.tabs(["Prediksi Teks", "Prediksi Massal", "Info Model"])

with tab_single:
    text = st.text_area("Masukkan teks atau tweet", height=140,
                        placeholder="Contoh: situs slot ini bikin rugi banyak orang")
    if st.button("Analisis Sentimen", use_container_width=True):
        if not text.strip():
            st.warning("Teks masih kosong.")
        else:
            siap, labels, proba, classes = predict([text], pipeline, kamus)
            label = labels[0]
            confidence = proba[0].max() * 100

            c1, c2 = st.columns(2)
            c1.metric("Prediksi Sentimen", str(label).capitalize())
            c2.metric("Confidence", f"{confidence:.2f}%")

            dist = pd.DataFrame({'Kelas': [str(c) for c in classes], 'Probabilitas': proba[0]})
            fig = px.bar(dist, x='Kelas', y='Probabilitas', color='Kelas',
                         color_discrete_map=COLORS, range_y=[0, 1])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'},
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Lihat hasil preprocessing"):
                st.code(siap[0] or "(kosong setelah preprocessing)")

with tab_batch:
    uploaded = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        col = st.selectbox("Pilih kolom teks", df.columns,
                           index=list(df.columns).index('full_text') if 'full_text' in df.columns else 0)
        if st.button("Jalankan Prediksi", use_container_width=True):
            texts = df[col].fillna('').astype(str).tolist()
            siap, labels, proba, classes = predict(texts, pipeline, kamus)
            df['prediksi'] = labels
            df['confidence'] = proba.max(axis=1).round(4)

            counts = pd.Series(labels).value_counts()
            fig = px.pie(values=counts.values, names=[str(i) for i in counts.index], hole=0.4,
                         color=[str(i) for i in counts.index], color_discrete_map=COLORS)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df[[col, 'prediksi', 'confidence']], use_container_width=True)
            st.download_button("Unduh Hasil", df.to_csv(index=False).encode('utf-8'),
                               file_name='hasil_prediksi.csv', mime='text/csv',
                               use_container_width=True)

with tab_model:
    le = pipeline['label_encoder']
    info = {
        'Algoritma': type(pipeline['model']).__name__,
        'Jumlah fitur TF-IDF': pipeline['tfidf'].max_features,
        'N-gram': pipeline['tfidf'].ngram_range,
        'Kelas': ', '.join(str(c) for c in le.classes_),
    }
    st.table(pd.DataFrame(info.items(), columns=['Atribut', 'Nilai']))
