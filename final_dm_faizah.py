# =========================================================
# STREAMLIT APP
# ANALISIS DATA MOTOR BEKAS
# BAGIAN A: CATBOOST (KLASIFIKASI)
# BAGIAN B: BAGGING REGRESSOR (REGRESI)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score

from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Analisis Motor Bekas",
    page_icon="🏍",
    layout="wide"
)

st.title("🏍 Analisis Motor Bekas (Klasifikasi & Regresi)")
st.caption("Upload dataset CSV – Abaikan warning kecil dari Streamlit")

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
    # ======================= BAGIAN A =====================
    # CATBOOST - KLASIFIKASI HARGA (BINARY)
    # =====================================================
    st.header("🅰 Bagian A – Klasifikasi Harga Motor (Binary CatBoost)")

    # =====================================================
    # TARGET BINARY (WAJIB)
    # =====================================================
    df["kategori_harga_binary"] = (
        df["harga"] > df["harga"].median()
    ).astype(int)

    # =====================================================
    # VISUALISASI TARGET
    # =====================================================
    fig, ax = plt.subplots()
    df["kategori_harga_binary"].value_counts().sort_index().plot(
        kind="bar", ax=ax
    )
    ax.set_xticklabels(["Murah", "Mahal"], rotation=0)
    ax.set_title("Distribusi Kategori Harga (Binary)")
    ax.set_ylabel("Jumlah Motor")
    st.pyplot(fig)

    # =====================================================
    # FEATURE & TARGET
    # =====================================================
    X_A = df.drop(
        ["harga", "kategori_harga", "kategori_bbm", "kategori_harga_binary"],
        axis=1
    )
    y_A = df["kategori_harga_binary"]

    # =====================================================
    # SPLIT DATA
    # =====================================================
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
        X_A, y_A,
        test_size=0.2,
        random_state=42,
        stratify=y_A
    )

    # =====================================================
    # SCALING
    # =====================================================
    scaler_A = StandardScaler()
    X_train_A = scaler_A.fit_transform(X_train_A)
    X_test_A = scaler_A.transform(X_test_A)

    # =====================================================
    # MODEL CATBOOST (BINARY)
    # =====================================================
    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_state=42,
        verbose=False
    )

    # =====================================================
    # TRAIN MODEL
    # =====================================================
    cat_model.fit(X_train_A, y_train_A)

    # =====================================================
    # PREDIKSI & EVALUASI
    # =====================================================
    y_pred_A = cat_model.predict(X_test_A)

    st.subheader("📊 Evaluasi CatBoost (Binary)")
    st.write("Accuracy:", round(accuracy_score(y_test_A, y_pred_A), 3))
    st.text(classification_report(y_test_A, y_pred_A))

    cm = confusion_matrix(y_test_A, y_pred_A)
    fig_cm, ax_cm = plt.subplots()
    ax_cm.imshow(cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # =====================================================
    # INPUT USER – BAGIAN A
    # =====================================================
    st.subheader("🔍 Prediksi Harga Motor (Input User)")

    input_A = {}
    for col in X_A.columns:
        input_A[col] = st.number_input(
            f"{col}",
            value=float(df[col].median())
        )

    if st.button("Prediksi Harga Motor"):
        input_df = pd.DataFrame([input_A])
        input_scaled = scaler_A.transform(input_df)
        pred = cat_model.predict(input_scaled)[0]

        hasil = "MAHAL 💸" if pred == 1 else "MURAH 💰"
        st.success(f"Kategori Harga Motor: {hasil}")

    st.caption(
        "ℹ️ Kategori MAHAL jika harga di atas median dataset, "
        "MURAH jika di bawah median."
    )


    # =====================================================
    # ======================= BAGIAN B =====================
    # BAGGING REGRESSOR – KONSUMSI BBM
    # =====================================================
    st.header("🅱 Bagian B – Prediksi Konsumsi BBM (Bagging Regressor)")

    X_B = df.drop(["konsumsiBBM"], axis=1)
    y_B = df["konsumsiBBM"]

    scaler_B = StandardScaler()
    X_B_scaled = scaler_B.fit_transform(X_B)

    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
        X_B_scaled, y_B, test_size=0.2, random_state=42
    )

    bagging = BaggingRegressor(
        estimator=LinearRegression(),
        n_estimators=50,
        random_state=42
    )

    bagging.fit(X_train_B, y_train_B)
    y_pred_B = bagging.predict(X_test_B)

    st.subheader("📊 Evaluasi Regresi Konsumsi BBM")
    st.write("R² Score :", round(r2_score(y_test_B, y_pred_B), 3))
    st.write("MAE      :", round(mean_absolute_error(y_test_B, y_pred_B), 2))

    # =====================================================
    # BATAS SEGMENTASI BBM
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
    # =====================================================
    st.subheader("⛽ Prediksi Konsumsi BBM Motor")

    input_B = {}
    for i, col in enumerate(X_B.columns):
        input_B[col] = st.number_input(
            label=f"Input {col}",
            value=float(df[col].median()),
            key=f"B_{i}_{col}"
        )

    if st.button("⛽ Prediksi Konsumsi BBM"):
        input_df_B = pd.DataFrame([input_B])
        input_scaled = scaler_B.transform(input_df_B)
        pred_bbm = bagging.predict(input_scaled)[0]

        st.success(
            f"🔋 Kategori Konsumsi BBM: {kategori_bbm(pred_bbm)}\n"
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