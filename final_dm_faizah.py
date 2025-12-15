# =====================================================
# STREAMLIT APP
# KLASIFIKASI + REGRESI + ENSEMBLE
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.impute import SimpleImputer

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="Analisis Penjualan", layout="wide")
st.title(" Analisis Penjualan: Klasifikasi & Regresi + Ensemble")

# =====================================================
# UPLOAD DATASET
# =====================================================
uploaded_file = st.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("⬆Silakan upload file CSV untuk memulai analisis")
    st.stop()

# =====================================================
# LOAD DATA (AMAN)
# =====================================================
df = pd.read_csv(uploaded_file, sep=None, engine="python")
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

st.subheader("Data Awal")
st.dataframe(df.head())

# =====================================================
# DATA CLEANING
# =====================================================
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.drop_duplicates()

num_cols = ["Units_Sold", "Unit_Price", "Revenue"]
for col in num_cols:
    df[col] = (
        df[col].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# FEATURE ENGINEERING DATE
df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"]   = df["Date"].dt.day
df = df.drop(columns=["Date"])

st.success("Data cleaning & feature engineering selesai")

# =====================================================
# ================= KLASIFIKASI =======================
# =====================================================
st.header(" Klasifikasi Revenue")

df["Revenue_Class"] = pd.qcut(
    df["Revenue"],
    q=3,
    labels=[0, 1, 2]
)

X_cls = df.drop(columns=["Revenue", "Revenue_Class"])
y_cls = df["Revenue_Class"]

if "Transaction_ID" in X_cls.columns:
    X_cls = X_cls.drop(columns=["Transaction_ID"])

cat_features_cls = X_cls.select_dtypes(include="object").columns.tolist()

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cls, y_cls, test_size=0.25, random_state=42, stratify=y_cls
)

cls_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    verbose=0
)

cls_model.fit(Xc_train, yc_train, cat_features=cat_features_cls)
pred_cls = cls_model.predict(Xc_test)

acc = accuracy_score(yc_test, pred_cls)
st.write("### Accuracy Klasifikasi:", round(acc, 4))

st.text("Classification Report")
st.text(classification_report(yc_test, pred_cls))

hasil_klasifikasi = pd.DataFrame({
    "Actual_Class": yc_test.values,
    "Predicted_Class": pred_cls.flatten()
})

st.download_button(
    "⬇️ Download Hasil Klasifikasi (CSV)",
    hasil_klasifikasi.to_csv(index=False),
    file_name="hasil_klasifikasi.csv",
    mime="text/csv"
)

# =====================================================
# ================= REGRESI ===========================
# =====================================================
st.header("Regresi Revenue + Ensemble")

X_reg = df.drop(columns=["Revenue", "Revenue_Class"])
y_reg = df["Revenue"]

if "Transaction_ID" in X_reg.columns:
    X_reg = X_reg.drop(columns=["Transaction_ID"])

# ---- CatBoost Regressor ----
cat_features_reg = X_reg.select_dtypes(include="object").columns.tolist()

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=42
)

cat_reg = CatBoostRegressor(
    iterations=400,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    verbose=0
)

cat_reg.fit(Xr_train, yr_train, cat_features=cat_features_reg)
pred_cat = cat_reg.predict(Xr_test)

# ---- Linear, Ridge, Lasso ----
X_enc = X_reg.copy()

for col in X_enc.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X_enc[col] = le.fit_transform(X_enc[col].astype(str))

imputer = SimpleImputer(strategy="median")
X_enc = pd.DataFrame(
    imputer.fit_transform(X_enc),
    columns=X_enc.columns
)

mask = ~y_reg.isna()
X_enc = X_enc.loc[mask]
y_clean = y_reg.loc[mask]

Xe_train, Xe_test, ye_train, ye_test = train_test_split(
    X_enc, y_clean, test_size=0.25, random_state=42
)

lr = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.001)

lr.fit(Xe_train, ye_train)
ridge.fit(Xe_train, ye_train)
lasso.fit(Xe_train, ye_train)

pred_lr    = lr.predict(Xe_test)
pred_ridge = ridge.predict(Xe_test)
pred_lasso = lasso.predict(Xe_test)

# ---- Ensemble Averaging ----
ensemble_pred = (
    pred_cat +
    pred_lr +
    pred_ridge +
    pred_lasso
) / 4

# =====================================================
# EVALUASI REGRESI
# =====================================================
def eval_reg(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

st.subheader("Evaluasi Regresi")

st.write("CatBoost Regressor:", eval_reg(yr_test, pred_cat))
st.write("Ensemble Regression:", eval_reg(ye_test, ensemble_pred))

hasil_regresi = pd.DataFrame({
    "Actual_Revenue": ye_test.values,
    "Pred_CatBoost": pred_cat,
    "Pred_Linear": pred_lr,
    "Pred_Ridge": pred_ridge,
    "Pred_Lasso": pred_lasso,
    "Pred_Ensemble": ensemble_pred
})

st.download_button(
    "Download Hasil Regresi (CSV)",
    hasil_regresi.to_csv(index=False),
    file_name="hasil_regresi_ensemble.csv",
    mime="text/csv"
)

st.success("Analisis selesai! Semua model berhasil dijalankan.")
