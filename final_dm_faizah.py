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

st.title("🏍 Analisis Motor Bekas (Klasifikasi & Regresi)")

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
        st.success(f"💰 Kategori Harga: **{label_map[pred]}**")

    # =====================================================
    # ======================= BAGIAN B =====================
    # REGRESI HARGA (RIDGE & LASSO)
    # =====================================================
    st.header("🅱 Bagian B – Regresi Harga")

    X_R = df.drop("harga", axis=1)
    y_R = df["harga"]

    X_train_R, X_test_R, y_train_R, y_test_R = train_test_split(
        X_R, y_R, test_size=0.2, random_state=42
    )

    scaler_R = StandardScaler()
    X_train_R = scaler_R.fit_transform(X_train_R)
    X_test_R = scaler_R.transform(X_test_R)

    ridge = Ridge()
    lasso = Lasso()

    models = {
        "Ridge Regression": ridge,
        "Lasso Regression": lasso
    }

    st.subheader("📊 Evaluasi Model Regresi")
    for name, model in models.items():
        model.fit(X_train_R, y_train_R)
        y_pred = model.predict(X_test_R)

        st.write(name)
        st.write("R² :", round(r2_score(y_test_R, y_pred), 3))
        st.write("MAE:", round(mean_absolute_error(y_test_R, y_pred), 2))

    # =====================================================
    # INPUT USER – BAGIAN B
    # =====================================================
    st.subheader("💸 Prediksi Harga Motor (Input User)")

    input_B = {}
    for col in X_R.columns:
        input_B[col] = st.number_input(
            f"{col}",
            float(df[col].median())
        )

    if st.button("Prediksi Harga Motor"):
        input_df = pd.DataFrame([input_B])
        input_scaled = scaler_R.transform(input_df)
        pred_price = ridge.predict(input_scaled)[0]

        st.success(f"💰 Estimasi Harga Motor: **Rp {pred_price:,.0f}**")
