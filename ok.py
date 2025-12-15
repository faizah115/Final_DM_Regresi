import pandas as pd

# Baca file CSV
df = pd.read_csv("DatasetPenjualanToko.csv")

# Ambil 4000 baris pertama
df_4000 = df.head(4000)

# Simpan ke file CSV baru
df_4000.to_csv("DatasetPenjualanToko_4000.csv", index=False)

print("Berhasil! File baru berisi", len(df_4000), "baris")
