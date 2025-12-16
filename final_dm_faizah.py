# =========================================================
# STREAMLIT APP
# ANALISIS DATA MOTOR BEKAS
# BAGIAN A: SVM (KLASIFIKASI)
# BAGIAN B: RIDGE REGRESSION (REGRESI)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# === MODEL ===
from sklearn.svm import SVC
from sklearn.linear_model import Ridge

# === METRIK ===
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score
)

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Analisis Motor Bekas",
    page_icon="🏍",
    layout="wide"
)

st.title("🏍 Analisis Data Motor Bekas")
st.caption("SVM untuk Klasifikasi & Ridge Regression untuk Prediksi BBM")

# =========================================================
# UPLOAD DATASET
# =========================================================
uploaded = st.file_uploader(
    "📂 Upload dataset motor_second_dataset.csv",
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

    # =====================================================
    # FEATURE ENGINEERING (TARGET)
    # =====================================================
    df["kategori_harga"] = pd.qcut(
        df["harga"], q=3, labels=["Rendah", "Sedang", "Tinggi"]
    )

    # =====================================================
    # ENCODING
    # =====================================================
    encoder = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    df["kategori_harga"] = LabelEncoder().fit_transform(df["kategori_harga"])

    # =====================================================
    # ======================= BAGIAN A =====================
    # SVM - KLASIFIKASI KATEGORI HARGA
    # =====================================================
    st.header("🅰 Bagian A – Klasifikasi Harga (SVM)")

    # Visualisasi
    fig, ax = plt.subplots()
    df["kategori_harga"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xticklabels(["Rendah", "Sedang", "Tinggi"], rotation=0)
    ax.set_title("Distribusi Kategori Harga")
    st.pyplot(fig)

    # Feature & Target
    X_A = df.drop(["harga", "kategori_harga", "konsumsiBBM"], axis=1)
    y_A = df["kategori_harga"]

    # Split
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
        X_A, y_A, test_size=0.2, random_state=42, stratify=y_A
    )

    # Scaling
    scaler_A = StandardScaler()
    X_train_A = scaler_A.fit_transform(X_train_A)
    X_test_A = scaler_A.transform(X_test_A)

    # Model SVM
    svm = SVC(kernel="rbf", C=1, gamma="scale")
    svm.fit(X_train_A, y_train_A)

    # Evaluasi
    y_pred_A = svm.predict(X_test_A)

    st.subheader("📊 Evaluasi SVM")
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
        pred = svm.predict(input_scaled)[0]

        label_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
        st.success(f"💰 Kategori Harga: *{label_map[pred]}*")

    # =====================================================
# ======================= BAGIAN B =====================
# BAGGING REGRESSOR - PREDIKSI KONSUMSI BBM
# =====================================================
st.header("🅱 Bagian B – Prediksi Konsumsi BBM (Bagging Regressor)")

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

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
# Scaling
# -----------------------
scaler_B = StandardScaler()
X_train_B = scaler_B.fit_transform(X_train_B)
X_test_B = scaler_B.transform(X_test_B)

# -----------------------
# Model Bagging Regressor
# -----------------------
bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=100,
    random_state=42
)

bagging.fit(X_train_B, y_train_B)

# -----------------------
# Evaluasi Model
# -----------------------
y_pred_B = bagging.predict(X_test_B)

st.subheader("📊 Evaluasi Bagging Regressor")
st.write("R² Score :", round(r2_score(y_test_B, y_pred_B), 3))
st.write("MAE      :", round(mean_absolute_error(y_test_B, y_pred_B), 2))

# =====================================================
# INPUT USER – BAGIAN B
# =====================================================
st.subheader("⛽ Prediksi Konsumsi BBM Motor")

input_B = {}
for i, col in enumerate(X_B.columns):
    input_B[col] = st.number_input(
        f"Input {col}",
        float(df[col].median()),
        key=f"B_{i}"
    )

if st.button("⛽ Prediksi Konsumsi BBM"):
    input_df_B = pd.DataFrame([input_B])
    input_scaled_B = scaler_B.transform(input_df_B)
    pred_bbm = bagging.predict(input_scaled_B)[0]

    st.success(f"🔋 Estimasi Konsumsi BBM: *{pred_bbm:.2f} km/l*")
