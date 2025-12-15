# =====================================================
# STREAMLIT APP
# KLASIFIKASI + REGRESI + ENSEMBLE (CATBOOST)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

st.set_page_config(
    page_title="Analisis Tiket Pesawat",
    layout="wide"
)

st.title("Analisis Klasifikasi & Regresi Harga Tiket Pesawat")
st.write("CatBoost • Regresi • Ensemble Method")

# =====================================================
# UPLOAD DATASET
# =====================================================
uploaded_file = st.file_uploader(
    "Upload dataset CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV untuk memulai analisis")

else:
    # =====================================================
    # LOAD DATA
    # =====================================================
    df = pd.read_csv(uploaded_file)

    st.subheader(" Data Awal")
    st.dataframe(df.head())

    st.write("Jumlah record:", df.shape[0])
    st.write("Jumlah atribut:", df.shape[1])

    # =====================================================
    # DATA CLEANING
    # =====================================================
    st.subheader(" Data Cleaning")

    st.write("Missing value per kolom:")
    st.dataframe(df.isnull().sum())

    st.write("Jumlah data duplikat:", df.duplicated().sum())

    df = df.drop_duplicates()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df_cleaned = df.copy()
    df_cleaned.to_csv("data_cleaned.csv", index=False)

    st.success("Data cleaned berhasil")

    # =====================================================
    # FEATURE ENGINEERING DATE
    # =====================================================
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df = df.drop(columns=["Date"])

    # =====================================================
    # KLASIFIKASI
    # =====================================================
    st.subheader("Klasifikasi Harga Tiket (CatBoost)")

    df["Total_Class"] = pd.qcut(
        df["Total"], q=3, labels=[0, 1, 2]
    )

    df_class = df.drop(columns=["Total"])
    df_class.to_csv("data_preprocessed_classification.csv", index=False)

    X = df_class.drop(columns=["Total_Class"])
    y = df_class["Total_Class"]

    cat_features = ["City", "Gender", "Airline", "Payment_Method"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model_cls = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        loss_function="MultiClass",
        verbose=0
    )

    model_cls.fit(X_train, y_train, cat_features=cat_features)
    y_pred = model_cls.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # Feature Importance
    fi = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model_cls.get_feature_importance()
    }).sort_values(by="Importance", ascending=False)

    fi.to_csv("feature_importance_classification.csv", index=False)

    fig_fi, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=fi, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig_fi)

    # =====================================================
    # REGRESI + ENSEMBLE
    # =====================================================
    st.subheader("Regresi & Ensemble Method")

    X_reg = df.drop(columns=["Total_Class", "Total"])
    y_reg = df["Total"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.25, random_state=42
    )

    # CatBoost Regressor
    cat_reg = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        verbose=0
    )

    cat_reg.fit(X_train, y_train, cat_features=cat_features)
    pred_cat = cat_reg.predict(X_test)

    # Encode untuk Linear Models
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    for col in cat_features:
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train_enc[col])
        X_test_enc[col] = le.transform(X_test_enc[col])

    lr = LinearRegression()
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.001)

    lr.fit(X_train_enc, y_train)
    ridge.fit(X_train_enc, y_train)
    lasso.fit(X_train_enc, y_train)

    pred_lr = lr.predict(X_test_enc)
    pred_ridge = ridge.predict(X_test_enc)
    pred_lasso = lasso.predict(X_test_enc)

    # Ensemble
    ensemble_pred = (
        pred_cat + pred_lr + pred_ridge + pred_lasso
    ) / 4

    # Evaluasi
    def eval_reg(y_true, y_pred):
        return {
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }

    st.write("CatBoost:", eval_reg(y_test, pred_cat))
    st.write("Ensemble:", eval_reg(y_test, ensemble_pred))

    # Simpan hasil
    reg_result = pd.DataFrame({
        "Actual_Total": y_test.values,
        "Pred_CatBoost": pred_cat,
        "Pred_Linear": pred_lr,
        "Pred_Ridge": pred_ridge,
        "Pred_Lasso": pred_lasso,
        "Pred_Ensemble": ensemble_pred
    })

    reg_result.to_csv("hasil_regresi_ensemble.csv", index=False)

    # Visualisasi
    fig_reg, ax = plt.subplots()
    ax.scatter(y_test, ensemble_pred, alpha=0.6)
    ax.set_xlabel("Actual Total")
    ax.set_ylabel("Predicted Total")
    ax.set_title("Actual vs Predicted - Ensemble Regression")
    st.pyplot(fig_reg)

    st.success("Proses selesai. File CSV tersimpan.")
