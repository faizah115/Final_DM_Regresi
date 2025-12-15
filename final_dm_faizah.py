# =========================================================
# STREAMLIT APP
# ANALISIS DATA MOTOR BEKAS
# BAGIAN A: KNN
# BAGIAN B: REGRESI ENSEMBLE (RIDGE & LASSO)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Analisis Motor Bekas",
    page_icon="🏍️",
    layout="wide"
)

st.title("🏍️ Analisis Motor Bekas (Klasifikasi & Regresi)")

# =========================================================
# UPLOAD DATASET
# =========================================================
uploaded = st.file_uploader(
    "📂 Upload dataset motor_second_dataset.csv",
    type=["csv"]
)

if uploaded:

    # =====================================================
    # 2. LOAD & EKSPLORASI DATA
    # =====================================================
    df = pd.read_csv(uploaded)

    st.subheader("📌 Pratinjau Dataset")
    st.dataframe(df, use_container_width=True)

    st.write("Jumlah baris:", df.shape[0])
    st.write("Jumlah kolom:", df.shape[1])
    st.write("Nama kolom:", df.columns.tolist())

    # =====================================================
    # 3. DATA CLEANING
    # =====================================================
    st.subheader("🧹 Data Cleaning")
    st.write("Missing value per kolom:")
    st.write(df.isnull().sum())

    df = df.dropna().drop_duplicates()
    st.write("Ukuran data setelah cleaning:", df.shape)

    # =====================================================
    # 4. FEATURE ENGINEERING (TARGET)
    # =====================================================
    df["kategori_harga"] = pd.qcut(
        df["harga"], q=3, labels=["Rendah", "Sedang", "Tinggi"]
    )

    df["kategori_bbm"] = pd.qcut(
        df["konsumsiBBM"], q=3, labels=["Boros", "Sedang", "Hemat"]
    )

    # =====================================================
    # 5. ENCODING DATA
    # =====================================================
    encoder = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    df["kategori_harga"] = LabelEncoder().fit_transform(df["kategori_harga"])
    df["kategori_bbm"] = LabelEncoder().fit_transform(df["kategori_bbm"])

    # =====================================================
    # ======================= BAGIAN A =====================
    # KLASIFIKASI KNN (KATEGORI HARGA)
    # =====================================================
    st.header("🅰️ Bagian A – Klasifikasi Harga (KNN)")

    fig, ax = plt.subplots()
    df["kategori_harga"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Segmentasi Motor Berdasarkan Kategori Harga")
    ax.set_xlabel("Kategori Harga")
    ax.set_ylabel("Jumlah Motor")
    ax.set_xticklabels(["Rendah", "Sedang", "Tinggi"], rotation=0)
    st.pyplot(fig)

    X_A = df.drop(["harga", "kategori_harga", "kategori_bbm"], axis=1)
    y_A = df["kategori_harga"]

    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
        X_A, y_A, test_size=0.2, random_state=42
    )

    scaler_A = StandardScaler()
    X_train_A_scaled = scaler_A.fit_transform(X_train_A)
    X_test_A_scaled = scaler_A.transform(X_test_A)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_A_scaled, y_train_A)
    y_pred_A = knn.predict(X_test_A_scaled)

    st.subheader("📊 Evaluasi KNN")
    st.write("Accuracy:", accuracy_score(y_test_A, y_pred_A))
    st.text(classification_report(y_test_A, y_pred_A))

    cm_A = confusion_matrix(y_test_A, y_pred_A)
    fig_cm, ax_cm = plt.subplots()
    ax_cm.imshow(cm_A)
    ax_cm.set_title("Confusion Matrix - KNN")
    st.pyplot(fig_cm)

# =====================================================
# INPUT USER – BAGIAN A
# PREDIKSI KATEGORI HARGA MOTOR BARU
# =====================================================
st.subheader("🔍 Prediksi Kategori Harga Motor (Input User)")

input_A = {}
for i, col in enumerate(X_A.columns):
    input_A[col] = st.number_input(
        label=f"Input {col}",
        value=float(df[col].median()),
        key=f"A_{i}_{col}"
    )

if st.button("🔍 Prediksi Kategori Harga"):
    input_df_A = pd.DataFrame([input_A])
    input_scaled_A = scaler_A.transform(input_df_A)
    pred_A = knn.predict(input_scaled_A)[0]

    label_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
    st.success(f"💰 Prediksi Kategori Harga: **{label_map[pred_A]}**")


# =====================================================
# ======================= BAGIAN B =====================
# REGRESI ENSEMBLE (RIDGE & LASSO)
# =====================================================
# =====================================================
# ======================= BAGIAN B =====================
# KLASIFIKASI KONSUMSI BBM (BOROS / SEDANG / HEMAT)
# =====================================================
st.header("🅱️ Bagian B – Klasifikasi Konsumsi BBM Motor")

# =====================================================
# VISUALISASI SEGMENTASI KONSUMSI BBM
# =====================================================
st.subheader("📊 Segmentasi Motor Berdasarkan Konsumsi BBM")

fig_bbm, ax_bbm = plt.subplots()
df["kategori_bbm"].value_counts().sort_index().plot(kind="bar", ax=ax_bbm)
ax_bbm.set_title("Segmentasi Konsumsi BBM Motor")
ax_bbm.set_xlabel("Kategori Konsumsi BBM")
ax_bbm.set_ylabel("Jumlah Motor")
ax_bbm.set_xticklabels(["Boros", "Sedang", "Hemat"], rotation=0)
st.pyplot(fig_bbm)

# =====================================================
# FEATURE & TARGET (KLASIFIKASI)
# =====================================================
fitur_bbm = [
    "tahun",
    "odometer",
    "mesin",
    "harga",
    "pajak"
]

X_B = df[fitur_bbm]
y_B = df["kategori_bbm"]   # target = Boros / Sedang / Hemat (encoded)

# =====================================================
# SPLIT DATA
# =====================================================
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X_B,
    y_B,
    test_size=0.2,
    random_state=42,
    stratify=y_B
)

# =====================================================
# SCALING
# =====================================================
scaler_B = StandardScaler()
X_train_B_scaled = scaler_B.fit_transform(X_train_B)
X_test_B_scaled  = scaler_B.transform(X_test_B)

# =====================================================
# MODEL KLASIFIKASI (KNN)
# =====================================================
knn_bbm = KNeighborsClassifier(n_neighbors=5)
knn_bbm.fit(X_train_B_scaled, y_train_B)

# =====================================================
# EVALUASI MODEL
# =====================================================
st.subheader("📊 Evaluasi Model Klasifikasi Konsumsi BBM")

y_pred_B = knn_bbm.predict(X_test_B_scaled)

st.write("Accuracy :", accuracy_score(y_test_B, y_pred_B))
st.text(classification_report(
    y_test_B,
    y_pred_B,
    target_names=["Boros", "Sedang", "Hemat"]
))

# =====================================================
# CONFUSION MATRIX
# =====================================================
cm_bbm = confusion_matrix(y_test_B, y_pred_B)

fig_cm_bbm, ax_cm_bbm = plt.subplots()
ax_cm_bbm.imshow(cm_bbm)
ax_cm_bbm.set_title("Confusion Matrix – Konsumsi BBM")
ax_cm_bbm.set_xlabel("Prediksi")
ax_cm_bbm.set_ylabel("Aktual")
ax_cm_bbm.set_xticks([0, 1, 2])
ax_cm_bbm.set_yticks([0, 1, 2])
ax_cm_bbm.set_xticklabels(["Boros", "Sedang", "Hemat"])
ax_cm_bbm.set_yticklabels(["Boros", "Sedang", "Hemat"])

for i in range(3):
    for j in range(3):
        ax_cm_bbm.text(j, i, cm_bbm[i, j], ha="center", va="center")

st.pyplot(fig_cm_bbm)

# =====================================================
# INPUT USER – OUTPUT KATEGORI BBM
# =====================================================
st.subheader("🔍 Prediksi Kategori Konsumsi BBM Motor")

input_B = {}
for i, col in enumerate(X_B.columns):
    input_B[col] = st.number_input(
        label=f"Input {col}",
        value=float(df[col].median()),
        key=f"B_{i}_{col}"
    )

if st.button("🔍 Prediksi Konsumsi BBM"):
    input_df_B = pd.DataFrame([input_B])
    input_scaled_B = scaler_B.transform(input_df_B)

    pred_B = knn_bbm.predict(input_scaled_B)[0]

    label_map = {
        0: "Boros",
        1: "Sedang",
        2: "Hemat"
    }

    st.success(
        f"⛽ Prediksi Konsumsi BBM Motor: **{label_map[pred_B]}**"
    )
