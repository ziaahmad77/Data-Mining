import pandas as pd

# 1. Baca file Excel
file_path = "data iklim harian - Semarang (2020-2023).xlsx"
df = pd.read_excel(file_path)

# 2. Cek kolom awal
print("Kolom awal:", df.columns.tolist())
print("Data 5 baris pertama:")
print(df.head())

# 3. Convert kolom Tanggal jadi datetime
df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce')

# 4. Cek missing values
print("\nJumlah missing value tiap kolom:")
print(df.isnull().sum())

# 5. Isi nilai kosong (contoh: pakai rata-rata untuk numerik)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].mean())

# 6. Normalisasi data numerik (opsional, contoh pakai Min-Max Scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 7. Simpan hasil preprocessing ke file baru
output_path = "data_preprocessed.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Preprocessing selesai! Data tersimpan di {output_path}")
