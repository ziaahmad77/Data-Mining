# ==============================
# KLASIFIKASI DECISION TREE
# DATA IKLIM SEMARANG 2020–2023
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# 5. Membuat model Decision Tree
dt = DecisionTreeClassifier(
    criterion='gini',     # bisa diganti 'entropy'
    max_depth=5,          # mencegah overfitting
    random_state=42
)

dt.fit(X_train, y_train)

# 6. Prediksi data testing
y_pred = dt.predict(X_test)

# 7. Evaluasi model
print("Akurasi Model Decision Tree :", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Contoh prediksi data baru
data_baru = [[28.5, 80, 5.2, 3.1]]
hasil = dt.predict(data_baru)

if hasil[0] == 1:
    print("\nPrediksi kondisi cuaca: HUJAN")
else:
    print("\nPrediksi kondisi cuaca: TIDAK HUJAN")
