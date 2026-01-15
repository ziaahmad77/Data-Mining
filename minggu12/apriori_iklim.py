import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Load dataset
data = pd.read_excel("data iklim harian - Semarang (2020-2023).xlsx")

# 2. Pilih kolom penting
df = data[['Tavg', 'RH_avg', 'RR', 'ss']].copy()

# 3. Handle missing value
df = df.fillna(df.mean())

# 4. Diskretisasi (ubah numerik -> kategori)
df['Suhu_Tinggi'] = df['Tavg'] >= 30
df['Suhu_Sedang'] = (df['Tavg'] >= 25) & (df['Tavg'] < 30)

df['Lembab_Tinggi'] = df['RH_avg'] >= 80
df['Hujan'] = df['RR'] > 0
df['Penyinaran_Tinggi'] = df['ss'] >= 5

# 5. Dataset transaksi (boolean)
transactions = df[
    ['Suhu_Tinggi', 'Suhu_Sedang', 'Lembab_Tinggi', 'Hujan', 'Penyinaran_Tinggi']
]

# 6. Frequent Itemset (Apriori)
frequent_itemsets = apriori(
    transactions,
    min_support=0.1,
    use_colnames=True
)

print("=== FREQUENT ITEMSETS ===")
print(frequent_itemsets.sort_values('support', ascending=False))

# 7. Association Rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

print("\n=== ASSOCIATION RULES ===")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

