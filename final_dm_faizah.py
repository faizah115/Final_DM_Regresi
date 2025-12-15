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
    page_icon="🏍",
    layout="wide"
)

st.title("🏍 Analisis Motor Bekas (Klasifikasi & Regresi) ")
st.title("BAIKAN EROR.. INI TIDAK EROR (LANGSUNG UP SAJA FILE CSV -NYA")

# =========================================================
# UPLOAD DATASET
# =========================================================
uploaded = st.file_uploader(
    "📂 Upload dataset motor_second_dataset.csv)",
    type=["csv"]
)

if uploaded:

    # =====================================================
    # LOAD DATA
    # =====================================================
    df = pd.read_csv(uploaded)

    st.subheader("📌 Pratinjau Dataset")
    st.dataframe(df, use_container_width=True)

    st.write("Jumlah baris:", df.shape[0])
    st.write("Jumlah kolom:", df.shape[1])
    st.write("Nama kolom:", df.columns.tolist())

    # =====================================================
    # DATA CLEANING
    # =====================================================
    st.subheader("🧹 Data Cleaning")
    st.write("Missing value per kolom:")
    st.write(df.isnull().sum())

    df = df.dropna().drop_duplicates()
    st.write("Ukuran data setelah cleaning:", df.shape)

    # =====================================================
    # FEATURE ENGINEERING (TARGET)
    # =====================================================
    df["kategori_harga"] = pd.qcut(
        df["harga"], q=3, labels=["Rendah", "Sedang", "Tinggi"]
    )

    df["kategori_bbm"] = pd.qcut(
        df["konsumsiBBM"], q=3, labels=["Boros", "Sedang", "Hemat"]
    )

    # =====================================================
    # ENCODING
    # =====================================================
    encoder = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    df["kategori_harga"] = LabelEncoder().fit_transform(df["kategori_harga"])
    df["kategori_bbm"] = LabelEncoder().fit_transform(df["kategori_bbm"])

    # =====================================================
    # ======================= BAGIAN A =====================
    # KNN - KLASIFIKASI KATEGORI HARGA
    # =====================================================
    st.header("🅰 Bagian A – Klasifikasi Harga (KNN)")

    fig, ax = plt.subplots()
    df["kategori_harga"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xticklabels(["Rendah", "Sedang", "Tinggi"], rotation=0)
    ax.set_title("Distribusi Kategori Harga")
    st.pyplot(fig)

    X_A = df.drop(["harga", "kategori_harga", "kategori_bbm"], axis=1)
    y_A = df["kategori_harga"]

    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
        X_A, y_A, test_size=0.2, random_state=42, stratify=y_A
    )

    scaler_A = StandardScaler()
    X_train_A = scaler_A.fit_transform(X_train_A)
    X_test_A = scaler_A.transform(X_test_A)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_A, y_train_A)
    y_pred_A = knn.predict(X_test_A)

    st.subheader("📊 Evaluasi KNN")
    st.write("Accuracy:", accuracy_score(y_test_A, y_pred_A))
    st.text(classification_report(y_test_A, y_pred_A))

    cm = confusion_matrix(y_test_A, y_pred_A)
    fig_cm, ax_cm = plt.subplots()
    ax_cm.imshow(cm)
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # =====================================================
    # INPUT USER – BAGIAN A
    # =====================================================
    st.subheader("🔍 Prediksi Kategori Harga Motor")

    input_A = {}
    for col in X_A.columns:
        input_A[col] = st.number_input(
            f"{col}",
            float(df[col].median())
        )

    if st.button("Prediksi Kategori Harga"):
        input_df = pd.DataFrame([input_A])
        input_scaled = scaler_A.transform(input_df)
        pred = knn.predict(input_scaled)[0]

        label_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
        st.success(f"💰 Kategori Harga: *{label_map[pred]}*")

    
  # =====================================================
# ======================= BAGIAN B =====================
# REGRESI KONSUMSI BBM (RANDOM FOREST)
# =====================================================
st.header("🅱 Bagian B – Prediksi Konsumsi BBM Motor")

from sklearn.ensemble import RandomForestRegressor

# -----------------------
# Feature & Target
# -----------------------
X_B = df.drop(["konsumsiBBM"], axis=1)
y_B = df["konsumsiBBM"]

# -----------------------
# Split Data
# -----------------------
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X_B, y_B, test_size=0.2, random_state=42
)

# -----------------------
# Model Random Forest
# -----------------------
rf_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_reg.fit(X_train_B, y_train_B)

# -----------------------
# Evaluasi Model
# -----------------------
y_pred_B = rf_reg.predict(X_test_B)

st.subheader("📊 Evaluasi Regresi Konsumsi BBM")
st.write("R² Score :", round(r2_score(y_test_B, y_pred_B), 3))
st.write("MAE      :", round(mean_absolute_error(y_test_B, y_pred_B), 2))

# =====================================================
# BATAS SEGMENTASI KONSUMSI BBM (BERDASARKAN DATA)
# =====================================================
bbm_q1 = df["konsumsiBBM"].quantile(0.33)
bbm_q2 = df["konsumsiBBM"].quantile(0.66)

def kategori_bbm(nilai):
    if nilai <= bbm_q1:
        return "Boros"
    elif nilai <= bbm_q2:
        return "Sedang"
    else:
        return "Hemat"

# =====================================================
# INPUT USER – BAGIAN B
# PREDIKSI KONSUMSI BBM MOTOR BARU
# =====================================================
st.subheader("⛽ Prediksi Konsumsi BBM Motor (Input User)")

input_B = {}
for i, col in enumerate(X_B.columns):
    input_B[col] = st.number_input(
        label=f"Input {col}",
        value=float(df[col].median()),
        key=f"B_{i}_{col}"
    )

if st.button("⛽ Prediksi Konsumsi BBM"):
    input_df_B = pd.DataFrame([input_B])
    pred_bbm = rf_reg.predict(input_df_B)[0]
    kategori = kategori_bbm(pred_bbm)

    st.success(
        f"🔋 Kategori Konsumsi BBM: *{kategori}*\n"
        f"(Estimasi: {pred_bbm:.2f} km/l)"
    )

# =====================================================
# SEGMENTASI KONSUMSI BBM
# =====================================================
segment_labels = ["Boros", "Sedang", "Hemat"]

df_segment = pd.DataFrame({
    "Aktual_BBM": y_test_B.values,
    "Prediksi_BBM": y_pred_B
})

df_segment["Segment_BBM"] = pd.qcut(
    df_segment["Prediksi_BBM"],
    q=3,
    labels=segment_labels
)

st.subheader("📊 Segmentasi Motor Berdasarkan Konsumsi BBM")

fig1, ax1 = plt.subplots()
df_segment["Segment_BBM"].value_counts().reindex(segment_labels).plot(
    kind="bar", ax=ax1
)
ax1.set_xlabel("Segment Konsumsi BBM")
ax1.set_ylabel("Jumlah Motor")
ax1.set_title("Distribusi Segmentasi Konsumsi BBM Motor")
st.pyplot(fig1)