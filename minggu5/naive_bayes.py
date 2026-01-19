# ==============================
# KLASIFIKASI NAIVE BAYES
# DATA IKLIM SEMARANG 2020–2023
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_excel("data iklim harian - Semarang (2020-2023).xlsx")

# 2. Membuat label klasifikasi (Hujan / Tidak Hujan)
df['Hujan'] = df['RR'].apply(lambda x: 1 if x > 0 else 0)

# 3. Menentukan fitur dan target
X = df[['Tavg', 'RH_avg', 'ss', 'ff_avg']]
y = df['Hujan']

# 4. Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 5. Membuat dan melatih model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 6. Prediksi data testing
y_pred = model.predict(X_test)

# 7. Evaluasi model
print("Akurasi Model :", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Contoh prediksi data baru
data_baru = [[28.5, 80, 5.2, 3.1]]
hasil = model.predict(data_baru)

if hasil[0] == 1:
    print("\nPrediksi kondisi cuaca: HUJAN")
else:
    print("\nPrediksi kondisi cuaca: TIDAK HUJAN")

