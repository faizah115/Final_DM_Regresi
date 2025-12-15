# =========================================================
# STREAMLIT APP
# ANALISIS DATA MOTOR BEKAS
# BAGIAN A: KNN (KLASIFIKASI HARGA)
# BAGIAN B: RANDOM FOREST (REGRESI KONSUMSI BBM)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Analisis Motor Bekas",
    layout="wide"
)

st.title("Analisis Motor Bekas (Klasifikasi & Regresi)")

# =========================================================
# UPLOAD DATASET
# =========================================================
uploaded = st.file_uploader(
    "Upload dataset motor_second_dataset.csv",
    type=["csv"]
)

# =========================================================
# PROSES HANYA JALAN JIKA DATA ADA
# =========================================================
if uploaded is not None:

    # =====================================================
    # LOAD DATA
    # =====================================================
    df = pd.read_csv(uploaded)

    st.subheader("Pratinjau Dataset")
    st.dataframe(df.head(), use_container_width=True)

    st.write("Jumlah baris:", df.shape[0])
    st.write("Jumlah kolom:", df.shape[1])
    st.write("Nama kolom:", df.columns.tolist())

    # =====================================================
    # DATA CLEANING
    # =====================================================
    df = df.dropna().drop_duplicates()

    # =====================================================
    # FEATURE ENGINEERING TARGET
    # =====================================================
    df["kategori_harga"] = pd.qcut(
        df["harga"],
        q=3,
        labels=["Rendah", "Sedang", "Tinggi"]
    )

    df["kategori_bbm"] = pd.qcut(
        df["konsumsiBBM"],
        q=3,
        labels=["Boros", "Sedang", "Hemat"]
    )

    # =====================================================
    # ENCODING DATA KATEGORIK
    # =====================================================
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    df["kategori_harga"] = LabelEncoder().fit_transform(df["kategori_harga"])
    df["kategori_bbm"] = LabelEncoder().fit_transform(df["kategori_bbm"])

    # =====================================================
    # ======================= BAGIAN A =====================
    # KNN - KLASIFIKASI KATEGORI HARGA
    # =====================================================
    st.header("Bagian A – Klasifikasi Harga Motor (KNN)")

    # Visualisasi distribusi
    fig_a, ax_a = plt.subplots()
    df["kategori_harga"].value_counts().sort_index().plot(
        kind="bar", ax=ax_a
    )
    ax_a.set_xticklabels(["Rendah", "Sedang", "Tinggi"], rotation=0)
    ax_a.set_title("Distribusi Kategori Harga")
    st.pyplot(fig_a)

    # Feature & Target
    X_A = df.drop(["harga", "kategori_harga", "kategori_bbm"], axis=1)
    y_A = df["kategori_harga"]

    # Split
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
        X_A, y_A,
        test_size=0.2,
        random_state=42,
        stratify=y_A
    )

    # Scaling
    scaler_A = StandardScaler()
    X_train_A = scaler_A.fit_transform(X_train_A)
    X_test_A = scaler_A.transform(X_test_A)

    # Model KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_A, y_train_A)
    y_pred_A = knn.predict(X_test_A)

    # Evaluasi
    st.subheader("Evaluasi KNN")
    st.write("Accuracy:", accuracy_score(y_test_A, y_pred_A))
    st.text(classification_report(y_test_A, y_pred_A))

    cm = confusion_matrix(y_test_A, y_pred_A)
    fig_cm, ax_cm = plt.subplots()
    ax_cm.imshow(cm)
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # =====================================================
    # ======================= BAGIAN B =====================
    # RANDOM FOREST - REGRESI KONSUMSI BBM
    # =====================================================
    st.header("Bagian B – Prediksi Konsumsi BBM Motor")

    # Feature & Target
    X_B = df.drop(["konsumsiBBM"], axis=1)
    y_B = df["konsumsiBBM"]

    # Split
    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
        X_B, y_B,
        test_size=0.2,
        random_state=42
    )

    # Model Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    rf.fit(X_train_B, y_train_B)

    # Prediksi
    y_pred_B = rf.predict(X_test_B)

    # Evaluasi
    st.subheader("Evaluasi Regresi")
    st.write("R² Score :", round(r2_score(y_test_B, y_pred_B), 3))
    st.write("MAE      :", round(mean_absolute_error(y_test_B, y_pred_B), 2))

    # =====================================================
    # SEGMENTASI KONSUMSI BBM
    # =====================================================
    segment_labels = ["Boros", "Sedang", "Hemat"]

    df_segment = pd.DataFrame({
        "Prediksi_BBM": y_pred_B
    })

    df_segment["Segment"] = pd.qcut(
        df_segment["Prediksi_BBM"],
        q=3,
        labels=segment_labels
    )

    fig_b, ax_b = plt.subplots()
    df_segment["Segment"].value_counts().reindex(segment_labels).plot(
        kind="bar", ax=ax_b
    )
    ax_b.set_title("Segmentasi Konsumsi BBM Motor")
    ax_b.set_xlabel("Kategori BBM")
    ax_b.set_ylabel("Jumlah Motor")
    st.pyplot(fig_b)

else:
    st.info("Silakan upload dataset untuk memulai analisis")
