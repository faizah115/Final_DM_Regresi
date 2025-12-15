# =====================================================
# STREAMLIT APP
# KLASIFIKASI + REGRESI + ENSEMBLE + INPUT USER
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="Prediksi Penjualan", layout="wide")
st.title("📊 Aplikasi Prediksi Revenue Penjualan")

# =====================================================
# UPLOAD DATASET
# =====================================================
uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])
if uploaded_file is None:
    st.info("Silakan upload dataset CSV")
    st.stop()

# =====================================================
# LOAD & CLEAN DATA
# =====================================================
df = pd.read_csv(uploaded_file, sep=None, engine="python")
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.drop_duplicates()

num_cols = ["Units_Sold", "Unit_Price", "Revenue"]
for col in num_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "."),
        errors="coerce"
    )

df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"]   = df["Date"].dt.day
df = df.drop(columns=["Date"])

# =====================================================
# KLASIFIKASI SETUP
# =====================================================
df["Revenue_Class"] = pd.qcut(df["Revenue"], q=3, labels=[0,1,2])

label_map = {0:"Rendah", 1:"Sedang", 2:"Tinggi"}

X_cls = df.drop(columns=["Revenue", "Revenue_Class", "Transaction_ID"], errors="ignore")
y_cls = df["Revenue_Class"]

cat_features = X_cls.select_dtypes(include="object").columns.tolist()

Xc_train, _, yc_train, _ = train_test_split(
    X_cls, y_cls, test_size=0.25, random_state=42, stratify=y_cls
)

cls_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    verbose=0
)
cls_model.fit(Xc_train, yc_train, cat_features=cat_features)

# =====================================================
# REGRESI + ENSEMBLE SETUP
# =====================================================
X_reg = df.drop(columns=["Revenue", "Revenue_Class", "Transaction_ID"], errors="ignore")
y_reg = df["Revenue"]

cat_reg = CatBoostRegressor(
    iterations=400,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    verbose=0
)
cat_reg.fit(X_reg, y_reg, cat_features=cat_features)

# Encode for linear models
X_enc = X_reg.copy()
encoders = {}

for col in X_enc.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X_enc[col] = le.fit_transform(X_enc[col].astype(str))
    encoders[col] = le

imputer = SimpleImputer(strategy="median")
X_enc = pd.DataFrame(imputer.fit_transform(X_enc), columns=X_enc.columns)

lr = LinearRegression().fit(X_enc, y_reg)
ridge = Ridge(alpha=1.0).fit(X_enc, y_reg)
lasso = Lasso(alpha=0.001).fit(X_enc, y_reg)

# =====================================================
# ================= INPUT USER ========================
# =====================================================
st.sidebar.header("🧾 Input Data Penjualan")

product = st.sidebar.selectbox("Product Name", df["Product_Name"].unique())
category = st.sidebar.selectbox("Category", df["Category"].unique())
store = st.sidebar.selectbox("Store Location", df["Store_Location"].unique())
payment = st.sidebar.selectbox("Payment Method", df["Payment_Method"].unique())

units = st.sidebar.number_input("Units Sold", min_value=1, value=1)
price = st.sidebar.number_input("Unit Price", min_value=1.0, value=1000.0)

year  = st.sidebar.number_input("Year", value=2024)
month = st.sidebar.slider("Month", 1, 12, 1)
day   = st.sidebar.slider("Day", 1, 31, 1)

# =====================================================
# PREDIKSI
# =====================================================
if st.sidebar.button("🔮 Prediksi"):
    input_df = pd.DataFrame([{
        "Product_Name": product,
        "Category": category,
        "Store_Location": store,
        "Payment_Method": payment,
        "Units_Sold": units,
        "Unit_Price": price,
        "Year": year,
        "Month": month,
        "Day": day
    }])

    # --- Klasifikasi ---
    pred_class = cls_model.predict(input_df)[0][0]
    label = label_map[pred_class]

    # --- Regresi ---
    pred_cat = cat_reg.predict(input_df)[0]

    input_enc = input_df.copy()
    for col in encoders:
        input_enc[col] = encoders[col].transform(input_enc[col])

    input_enc = pd.DataFrame(imputer.transform(input_enc), columns=input_enc.columns)

    pred_lr = lr.predict(input_enc)[0]
    pred_ridge = ridge.predict(input_enc)[0]
    pred_lasso = lasso.predict(input_enc)[0]

    ensemble_pred = (pred_cat + pred_lr + pred_ridge + pred_lasso) / 4

    # =====================================================
    # OUTPUT
    # =====================================================
    st.success("🎉 Hasil Prediksi")

    st.write("### 🔖 Prediksi Kelas Revenue:")
    st.markdown(f"## **{label}**")

    st.write("### 💰 Prediksi Revenue (Ensemble):")
    st.markdown(f"## **Rp {ensemble_pred:,.2f}**")
