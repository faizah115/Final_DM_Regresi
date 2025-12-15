# =========================================================
# STREAMLIT APP
# ANALISIS DATA MOTOR BEKAS
# BAGIAN A: KLASIFIKASI KNN
# BAGIAN B: REGRESI ENSEMBLE
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

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
# UPLOAD DATA
# =========================================================
uploaded = st.file_uploader(
    "📂 Upload dataset motor_second.csv",
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

    # =====================================================
    # 3. DATA CLEANING
    # =====================================================
    st.subheader("🧹 Data Cleaning")
    st.write(df.isnull().sum())

    df = df.dropna()
    df = df.drop_duplicates()

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
    df["kategori_bbm"]   = LabelEncoder().fit_transform(df["kategori_bbm"])

    # =====================================================
    # ======================= BAGIAN A =====================
    # KLASIFIKASI KNN
    # =====================================================
    st.header("🅰️ Bagian A – Klasifikasi Harga (KNN)")

    # Visualisasi Segmentasi Harga
    fig, ax = plt.subplots()
    df["kategori_harga"].value_counts().sort_index().plot(
        kind="bar", ax=ax
    )
    ax.set_title("Segmentasi Motor Berdasarkan Kategori Harga")
    ax.set_xlabel("Kategori Harga")
    ax.set_ylabel("Jumlah Motor")
    ax.set_xticklabels(["Rendah", "Sedang", "Tinggi"], rotation=0)
    st.pyplot(fig)

    # Feature & Target
    X_A = df.drop(["harga", "kategori_harga", "kategori_bbm"], axis=1)
    y_A = df["kategori_harga"]

    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
        X_A, y_A, test_size=0.2, random_state=42
    )

    scaler_A = StandardScaler()
    X_train_A = scaler_A.fit_transform(X_train_A)
    X_test_A  = scaler_A.transform(X_test_A)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_A, y_train_A)

    # ================= INPUT USER (A) ====================
    st.subheader("🔍 Prediksi Kategori Harga Motor Baru")

    col1, col2 = st.columns(2)
    with col1:
        tahun = st.number_input("Tahun", 1990, 2025, 2018)
        odometer = st.number_input("Odometer", 0, 500000, 20000)
        mesin = st.number_input("Kapasitas Mesin (cc)", 100, 2000, 150)
    with col2:
        transmisi = st.number_input("Transmisi (encoded)", 0, 5, 1)
        jenis = st.number_input("Jenis Motor (encoded)", 0, 5, 1)
        pajak = st.number_input("Pajak (encoded)", 0, 1, 1)
        konsumsiBBM = st.number_input("Konsumsi BBM", 10, 100, 40)

    if st.button("🔍 Prediksi Harga"):
        input_A = np.array([[tahun, odometer, mesin, transmisi, jenis, pajak, konsumsiBBM]])
        input_A = scaler_A.transform(input_A)
        pred_harga = knn.predict(input_A)[0]

        label_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
        st.success(f"💰 Prediksi Kategori Harga: **{label_map[pred_harga]}**")

    # =====================================================
    # ======================= BAGIAN B =====================
    # REGRESI ENSEMBLE
    # =====================================================
    st.header("🅱️ Bagian B – Prediksi Konsumsi BBM (Regresi)")

    # Segmentasi Konsumsi BBM
    df["segmen_bbm"] = pd.qcut(
        df["konsumsiBBM"], q=3, labels=["Boros", "Sedang", "Hemat"]
    )

    fig2, ax2 = plt.subplots()
    df["segmen_bbm"].value_counts().plot(kind="bar", ax=ax2)
    ax2.set_title("Segmentasi Motor Berdasarkan Konsumsi BBM")
    ax2.set_xlabel("Segmen BBM")
    ax2.set_ylabel("Jumlah Motor")
    st.pyplot(fig2)

    # Feature & Target
    X_B = df.drop("konsumsiBBM", axis=1)
    y_B = df["konsumsiBBM"]

    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
        X_B, y_B, test_size=0.25, random_state=42
    )

    scaler_B = StandardScaler()
    X_train_B = scaler_B.fit_transform(X_train_B)
    X_test_B  = scaler_B.transform(X_test_B)

    models = {
        "SVR": SVR(),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "Ridge": Ridge()
    }

    st.subheader("📊 Evaluasi Regresi (R² & MAE)")
    for name, model in models.items():
        model.fit(X_train_B, y_train_B)
        y_pred = model.predict(X_test_B)

        r2 = r2_score(y_test_B, y_pred)
        mae = mean_absolute_error(y_test_B, y_pred)

        st.write(f"**{name}** → R²: {r2:.3f} | MAE: {mae:.2f}")
