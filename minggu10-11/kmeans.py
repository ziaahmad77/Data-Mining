import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
data = pd.read_excel("data iklim harian - Semarang (2020-2023).xlsx")

# 2. Pilih fitur untuk clustering
X = data[['Tavg', 'RH_avg', 'RR', 'ss']]

# 3. Tangani missing value
X = X.fillna(X.mean())

# 4. Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. K-Means clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. Visualisasi (2D)
plt.figure(figsize=(8, 6))
plt.scatter(
    data['Tavg'],
    data['RH_avg'],
    c=data['Cluster'],
    alpha=0.7
)
plt.xlabel('Suhu Rata-rata (Tavg)')
plt.ylabel('Kelembaban Rata-rata (RH_avg)')
plt.title('K-Means Clustering Data Iklim Harian Semarang')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# 7. Ringkasan cluster
print("Jumlah data tiap cluster:")
print(data['Cluster'].value_counts())

print("\nRata-rata tiap fitur per cluster:")
print(data.groupby('Cluster')[['Tavg', 'RH_avg', 'RR', 'ss']].mean())

