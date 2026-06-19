import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
import json
import os
import base64
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px

script_dir = os.path.dirname(os.path.abspath(__file__))

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    bin_str = get_base64('judol.png')
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.warning("Fail 'judol.png' tidak dijumpai. Latar belakang tidak dapat dimuatkan.")

def cleanTweets(text):
    text = re.sub(r'@\w+', '', str(text))
    text = re.sub('<USERNAME>', '', text)
    text = re.sub('https?:\/\/\S+', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text.lower().strip())
    text = re.sub('#', '', text)
    text = re.sub('\d+', '', text)
    text = re.sub('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]+', '', text)
    text = re.sub('RT[\s]+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text):
    return text.split() if isinstance(text, str) else []

def replace_abbreviations(words, abbreviation_dict):
    return [abbreviation_dict.get(word.lower(), word) for word in words]

def remove_stop_words(words, stop_words_set):
    return [word for word in words if word.lower() not in stop_words_set]

def custom_stemming(words, stem_dict):
    return [stem_dict.get(word.lower(), word) for word in words]

def load_local_json(filename):
    if filename and filename != "❌ Tidak Pakai Kamus":
        full_path = os.path.join(script_dir, filename)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None

st.set_page_config(page_title="Analisis Sentimen", page_icon="🎲", layout="wide")

def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css("style.css")

st.title("🛡️ Analisis Sentimen Pengguna X Terhadap Fenomena dan Dampak Sosial Judi Online di Indonesia 🛡️")

json_files = ["❌ Tidak Pakai Kamus"] + [f for f in os.listdir(script_dir) if f.endswith('.json')]

with st.sidebar:
    st.header("1. Input Data")
    uploaded_file = st.file_uploader("Pilih dataset (CSV/Excel)", type=["csv", "xlsx"])

    st.markdown("---")
    st.header("2. Pilih File Kamus")

    idx_norm = json_files.index('kamus_normalisasi.json') if 'kamus_normalisasi.json' in json_files else 0
    sel_norm = st.selectbox("Kamus Normalisasi:", json_files, index=idx_norm)

    idx_stop = json_files.index('kamus_stopword.json') if 'kamus_stopword.json' in json_files else 0
    sel_stop = st.selectbox("Kamus Stopword:", json_files, index=idx_stop)

    idx_stem = json_files.index('kamus_stemming.json') if 'kamus_stemming.json' in json_files else 0
    sel_stem = st.selectbox("Kamus Stemming:", json_files, index=idx_stem)

kamus_norm = load_local_json(sel_norm)
kamus_stem = load_local_json(sel_stem)
kamus_stopword_list = load_local_json(sel_stop)
kamus_stopword = set(kamus_stopword_list) if kamus_stopword_list else None

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    teks_col = 'full_text' if 'full_text' in df_raw.columns else df_raw.columns[0]

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Data Mentah", "⚙️ Proses Pipeline", "📊 Hasil EDA", "🤖 Evaluasi ML"])

    with tab1:
        st.subheader("1. Kondisi Data Awal")
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Data Awal", len(df_raw))
        c2.metric("Data Kosong", df_raw[teks_col].isnull().sum())
        c3.metric("Data Duplikat", df_raw.duplicated(subset=[teks_col]).sum())

        st.dataframe(df_raw.head(10), use_container_width=True)
        st.markdown("---")

        df_no_dup = df_raw.drop_duplicates(subset=[teks_col]).dropna(subset=[teks_col]).copy()

        st.subheader("2. Kondisi Setelah Drop Data Kotor")
        c4, c5, c6 = st.columns(3)
        c4.metric("Sisa Data Tersedia", len(df_no_dup), f"-{len(df_raw) - len(df_no_dup)} Baris Dibuang", delta_color="inverse")
        c5.metric("Data Kosong Sekarang", df_no_dup[teks_col].isnull().sum())
        c6.metric("Data Duplikat Sekarang", df_no_dup.duplicated(subset=[teks_col]).sum())

        st.dataframe(df_no_dup.head(10), use_container_width=True)

        st.markdown("---")
        st.subheader("3. Filter Data Berdasarkan Label Sentimen")

        if 'label' in df_no_dup.columns:
            unique_labels = df_no_dup['label'].dropna().unique().tolist()
            pilihan_label = ["Semua"] + unique_labels

            selected_label = st.selectbox("🔍 Tampilkan contoh data untuk label:", pilihan_label)

            if selected_label == "Semua":
                st.dataframe(df_no_dup, use_container_width=True)
            else:
                df_filtered = df_no_dup[df_no_dup['label'] == selected_label]
                st.caption(f"Menampilkan {len(df_filtered)} baris data dengan label **{selected_label}**:")
                st.dataframe(df_filtered, use_container_width=True)
        else:
            st.info("⚠️ Kolom 'label' tidak ditemukan pada dataset ini, sehingga fitur filter tidak dapat digunakan.")

    with tab2:
        st.subheader("Transformasi Data per Tahapan")
        if st.button("🚀 Jalankan Proses", use_container_width=True):
            with st.spinner("Memproses tahapan NLP..."):
                df = df_no_dup.copy()
                df['cleaning'] = df[teks_col].apply(cleanTweets)
                df['tokenizing'] = df['cleaning'].apply(tokenize)

                df['normalisasi'] = df['tokenizing'].apply(lambda x: replace_abbreviations(x, kamus_norm)) if kamus_norm else df['tokenizing']
                df['stopword'] = df['normalisasi'].apply(lambda x: remove_stop_words(x, kamus_stopword)) if kamus_stopword else df['normalisasi']
                df['stemming'] = df['stopword'].apply(lambda x: custom_stemming(x, kamus_stem)) if kamus_stem else df['stopword']

                df['TEXT_SIAP'] = df['stemming'].apply(lambda x: ' '.join(x))
                st.session_state['df_final'] = df
                st.success("✅ Tahapan Selesai!")

        if 'df_final' in st.session_state:
            df_view = st.session_state['df_final']
            st.markdown("#### A. Tahap Cleaning")
            st.dataframe(df_view[[teks_col, 'cleaning']].head(5), use_container_width=True)
            st.markdown("#### B. Tahap Tokenizing")
            st.dataframe(df_view[['cleaning', 'tokenizing']].head(5), use_container_width=True)
            st.markdown("#### C. Tahap Normalisasi")
            st.dataframe(df_view[['tokenizing', 'normalisasi']].head(5), use_container_width=True)
            st.markdown("#### D. Tahap Stopword Removal")
            st.dataframe(df_view[['normalisasi', 'stopword']].head(5), use_container_width=True)
            st.markdown("#### E. Tahap Stemming")
            st.dataframe(df_view[['stopword', 'stemming']].head(5), use_container_width=True)
            st.markdown("#### F. Hasil Akhir (TEXT SIAP)")
            st.dataframe(df_view[['stemming', 'TEXT_SIAP']].head(5), use_container_width=True)

            st.markdown("---")
            csv = df_view.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Seluruh Hasil CSV", data=csv, file_name='data_bersih_lengkap.csv', mime='text/csv')

    with tab3:
        if 'df_final' in st.session_state:
            df = st.session_state['df_final']
            all_text = ' '.join(df['TEXT_SIAP'].replace('', np.nan).dropna())

            if all_text.strip() != "":
                st.subheader("☁️ WordCloud Dominan")
                wc = WordCloud(width=800, height=400, background_color='rgba(0,0,0,0)', colormap='cool', mode='RGBA').generate(all_text)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                fig_wc.patch.set_alpha(0); ax_wc.imshow(wc, interpolation='bilinear'); ax_wc.axis('off')
                st.pyplot(fig_wc)

                st.subheader("📈 Analisis N-Gram")
                def plot_ngram(n_range, title, color):
                    try:
                        vec = CountVectorizer(ngram_range=(n_range, n_range), max_features=10).fit([all_text])
                        bag_of_words = vec.transform([all_text])
                        sum_words = bag_of_words.sum(axis=0)
                        words_freq = sorted([(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()], key=lambda x: x[1], reverse=True)
                        words, counts = zip(*words_freq)
                        fig = go.Figure(go.Bar(x=counts, y=words, orientation='h', marker_color=color))
                        fig.update_layout(title=title, yaxis={'autorange': 'reversed'}, height=400, paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
                        return fig
                    except:
                        return go.Figure().update_layout(title="Data tidak cukup", paper_bgcolor='rgba(0,0,0,0)')

                c1, c2, c3 = st.columns(3)
                with c1: st.plotly_chart(plot_ngram(1, "Unigram (1 Kata)", '#00C6FF'), use_container_width=True)
                with c2: st.plotly_chart(plot_ngram(2, "Bigram (2 Kata)", '#FF007F'), use_container_width=True)
                with c3: st.plotly_chart(plot_ngram(3, "Trigram (3 Kata)", '#7928CA'), use_container_width=True)

                if 'label' in df.columns:
                    st.subheader("📊 Distribusi Label Sentimen")
                    sent_counts = df['label'].value_counts()
                    fig_sent = go.Figure(go.Pie(labels=sent_counts.index, values=sent_counts.values, hole=.4, marker=dict(colors=['#FF5252', '#00E676', '#9E9E9E'])))
                    fig_sent.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
                    st.plotly_chart(fig_sent)
        else:
            st.info("Jalankan pipeline di tab 'Proses Pipeline' terlebih dahulu.")

    with tab4:
        st.subheader("🤖 Klasifikasi Sentimen - Perbandingan Model ML")

        if 'df_final' not in st.session_state:
            st.warning("⚠️ Jalankan Proses Pipeline di Tab 2 terlebih dahulu untuk mendapatkan 'TEXT_SIAP'.")
        elif 'label' not in st.session_state['df_final'].columns:
            st.error("❌ Dataset Anda tidak memiliki kolom 'label'. Proses Machine Learning dibatalkan.")
        else:
            st.info("Bandingkan performa berbagai model ML menggunakan ekstraksi fitur TF-IDF Unigram, Bigram, dan Trigram.")

            model_options = {
                "XGBoost": "xgboost",
                "Logistic Regression": "lr",
                "Naive Bayes": "nb",
                "Decision Tree": "dt",
                "Random Forest": "rf",
                "SVM (Linear)": "svm",
            }
            selected_model_name = st.selectbox(
                "🤖 Pilih Model Klasifikasi:",
                list(model_options.keys()),
                index=0,
                help="Pilih algoritma machine learning yang ingin digunakan untuk klasifikasi sentimen."
            )

            def get_model(name):
                if name == "XGBoost":
                    return xgb.XGBClassifier(
                        n_estimators=200, learning_rate=0.05, max_depth=4,
                        subsample=0.7, colsample_bytree=0.7,
                        eval_metric='mlogloss', random_state=42
                    )
                elif name == "Logistic Regression":
                    return LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
                elif name == "Naive Bayes":
                    return MultinomialNB(alpha=1.0)
                elif name == "Decision Tree":
                    return DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42)
                elif name == "Random Forest":
                    return RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
                elif name == "SVM (Linear)":
                    return LinearSVC(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)

            if st.button(f"⚙️ Mulai Training ML ({selected_model_name})", use_container_width=True):
                with st.spinner(f"Sedang melakukan training {selected_model_name} pada 3 skenario... ini mungkin memakan waktu sebentar."):
                    df_ml = st.session_state['df_final'].copy()
                    df_ml = df_ml.dropna(subset=['TEXT_SIAP', 'label'])

                    X = df_ml['TEXT_SIAP']
                    y = df_ml['label']

                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    target_names = [str(cls) for cls in le.classes_]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                    )

                    def train_evaluate(vectorizer, model):
                        X_train_vec = vectorizer.fit_transform(X_train)
                        X_test_vec = vectorizer.transform(X_test)
                        model.fit(X_train_vec, y_train)
                        y_pred = model.predict(X_test_vec)
                        acc = accuracy_score(y_test, y_pred)
                        cm = confusion_matrix(y_test, y_pred)
                        cr = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                        return acc, cm, cr

                    vec_tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=1000, min_df=2)
                    acc_tfidf, cm_tfidf, cr_tfidf = train_evaluate(vec_tfidf, get_model(selected_model_name))

                    vec_2gram = TfidfVectorizer(ngram_range=(2, 2), max_features=1000, min_df=2)
                    acc_2gram, cm_2gram, cr_2gram = train_evaluate(vec_2gram, get_model(selected_model_name))

                    vec_3gram = TfidfVectorizer(ngram_range=(3, 3), max_features=1000, min_df=2)
                    acc_3gram, cm_3gram, cr_3gram = train_evaluate(vec_3gram, get_model(selected_model_name))

                    st.success(f"✅ Training {selected_model_name} Selesai!")

                    st.markdown(f"#### 🏆 Perbandingan Akurasi — {selected_model_name}")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric(f"Unigram × {selected_model_name}", f"{acc_tfidf*100:.2f}%")
                    col_m2.metric(f"Bigram × {selected_model_name}", f"{acc_2gram*100:.2f}%")
                    col_m3.metric(f"Trigram × {selected_model_name}", f"{acc_3gram*100:.2f}%")

                    st.markdown("---")

                    st.markdown("#### 📑 Classification Report Lengkap")
                    st.markdown("<small>Melihat detail Precision, Recall, dan F1-Score per kelas sentimen.</small>", unsafe_allow_html=True)

                    def format_cr(cr_dict):
                        return pd.DataFrame(cr_dict).transpose().style.format(precision=3)

                    c_cr1, c_cr2, c_cr3 = st.columns(3)
                    with c_cr1:
                        st.markdown("**1. Laporan TF-IDF**")
                        st.dataframe(format_cr(cr_tfidf), use_container_width=True)
                    with c_cr2:
                        st.markdown("**2. Laporan 2-Gram**")
                        st.dataframe(format_cr(cr_2gram), use_container_width=True)
                    with c_cr3:
                        st.markdown("**3. Laporan 3-Gram**")
                        st.dataframe(format_cr(cr_3gram), use_container_width=True)

                    st.markdown("---")

                    st.markdown("#### 📉 Confusion Matrix")
                    def plot_cm(cm, title):
                        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Purples',
                                        x=target_names, y=target_names, title=title)
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'},
                                          xaxis_title="Prediksi", yaxis_title="Aktual")
                        return fig

                    c_cm1, c_cm2, c_cm3 = st.columns(3)
                    with c_cm1: st.plotly_chart(plot_cm(cm_tfidf, "CM: TF-IDF"), use_container_width=True)
                    with c_cm2: st.plotly_chart(plot_cm(cm_2gram, "CM: 2-Gram"), use_container_width=True)
                    with c_cm3: st.plotly_chart(plot_cm(cm_3gram, "CM: 3-Gram"), use_container_width=True)

else:
    st.info("👈 Harap unggah dataset untuk memulai analisis.")
