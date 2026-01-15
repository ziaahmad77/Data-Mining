import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load dataset
data = pd.read_excel("data iklim harian - Semarang (2020-2023).xlsx")

# 2. Pilih fitur (X) dan target (Y)
X = data[['RH_avg', 'RR', 'ss', 'ff_avg']]
y = data['Tavg']

# 3. Handle missing value
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# 4. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Buat dan latih model regresi
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Prediksi
y_pred = model.predict(X_test)

# 7. Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# 8. Visualisasi hasil
plt.scatter(y_test, y_pred)
plt.xlabel("Suhu Aktual (Tavg)")
plt.ylabel("Suhu Prediksi (Tavg)")
plt.title("Regresi Linier Prediksi Suhu Rata-rata")
plt.show()
