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
    # LOAD DATA
    # =====================================================
    df = pd.read_csv(uploaded)

    st.subheader("📌 Pratinjau Dataset")
    st.dataframe(df, use_container_width=True)

    # =====================================================
    # DATA CLEANING
    # =====================================================
    st.subheader("🧹 Data Cleaning")
    st.write(df.isnull().sum())

    df = df.dropna().drop_duplicates()
    st.write("Ukuran data setelah cleaning:", df.shape)

    # =====================================================
    # FEATURE ENGINEERING
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
    # ================= BAGIAN A ===========================
    # KLASIFIKASI KNN
    # =====================================================
    st.header("🅰️ Bagian A – Klasifikasi Harga (KNN)")

    fig, ax = plt.subplots()
    df["kategori_harga"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Segmentasi Motor Berdasarkan Kategori Harga")
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

    # ================= INPUT USER =========================
    st.subheader("🔍 Prediksi Kategori Harga Motor Baru")

    user_input = {}
    for col in X_A.columns:
        user_input[col] = st.number_input(f"Input {col}", value=float(df[col].median()))

    if st.button("🔍 Prediksi Harga"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler_A.transform(input_df)
        pred = knn.predict(input_scaled)[0]

        label_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
        st.success(f"💰 Prediksi Kategori Harga: **{label_map[pred]}**")

 # =====================================================
# ================= BAGIAN B ===========================
# REGRESI ENSEMBLE
# =====================================================
st.header("🅱️ Bagian B – Prediksi Konsumsi BBM (Regresi)")

# -----------------------
# Segmentasi Konsumsi BBM (Visualisasi)
# -----------------------
df["segmen_bbm"] = pd.qcut(
    df["konsumsiBBM"],
    q=3,
    labels=["Boros", "Sedang", "Hemat"]
)

fig2, ax2 = plt.subplots()
df["segmen_bbm"].value_counts().plot(kind="bar", ax=ax2)
ax2.set_title("Segmentasi Motor Berdasarkan Konsumsi BBM")
ax2.set_xlabel("Segmen BBM")
ax2.set_ylabel("Jumlah Motor")
st.pyplot(fig2)

# -----------------------
# Feature & Target (NUMERIK SAJA)
# -----------------------
X_B = df.drop(
    ["konsumsiBBM", "kategori_harga", "kategori_bbm", "segmen_bbm"],
    axis=1,
    errors="ignore"
)
y_B = df["konsumsiBBM"]

# -----------------------
# Split Data
# -----------------------
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X_B, y_B, test_size=0.25, random_state=42
)

# -----------------------
# Scaling
# -----------------------
scaler_B = StandardScaler()
X_train_B_scaled = scaler_B.fit_transform(X_train_B)
X_test_B_scaled = scaler_B.transform(X_test_B)

# -----------------------
# Model Regresi Ensemble
# -----------------------
models = {
    "SVR": SVR(),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Ridge": Ridge()
}

# -----------------------
# Evaluasi Model
# -----------------------
st.subheader("📊 Evaluasi Regresi (R² & MAE)")
for name, model in models.items():
    model.fit(X_train_B_scaled, y_train_B)
    y_pred = model.predict(X_test_B_scaled)

    r2 = r2_score(y_test_B, y_pred)
    mae = mean_absolute_error(y_test_B, y_pred)

    st.write(
        f"**{name}** → "
        f"R²: {r2:.3f} | "
        f"MAE: {mae:.2f}"
    )
# =====================================================
# INPUT USER – PREDIKSI KONSUMSI BBM (BAGIAN B)
# =====================================================
st.subheader("🔍 Prediksi Konsumsi BBM Motor Baru")

input_bbm = {}

for i, col in enumerate(X_B.columns):
    input_bbm[col] = st.number_input(
        label=f"Input {col}",
        value=float(df[col].median()),
        key=f"bbm_{i}_{col}"   # <<< KEY UNIK (INI PENTING)
    )

if st.button("🔍 Prediksi Konsumsi BBM"):
    input_df_B = pd.DataFrame([input_bbm])

    # Scaling
    input_scaled_B = scaler_B.transform(input_df_B)

    # Gunakan model utama (SVR)
    model_bbm = models["SVR"]
    bbm_pred = model_bbm.predict(input_scaled_B)[0]

    st.success(f"⛽ Prediksi Konsumsi BBM: **{bbm_pred:.2f}**")

    # Interpretasi bisnis
    if bbm_pred < df["konsumsiBBM"].quantile(0.33):
        kategori = "Boros"
    elif bbm_pred < df["konsumsiBBM"].quantile(0.66):
        kategori = "Sedang"
    else:
        kategori = "Hemat"

    st.info(
        f"📌 Interpretasi: Motor ini diperkirakan memiliki konsumsi BBM "
        f"**{kategori}**."
    )
