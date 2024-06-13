from lifetimes import BetaGeoFitter
import datetime as dt
import pandas as pd

# Adım 1: Veriyi yükleme ve ön işleme
df = pd.read_csv('train_BRCpofr.csv')

# Bugünün tarihi
today_date = dt.date.today()

# Assuming 'vintage' represents the number of months since the first purchase
# Convert 'vintage' from months to weeks and make sure it's in numeric format
df['T'] = pd.to_numeric(df['vintage']) * 4  # aylık olan vintage değerini haftaya dönüştürüyoruz
df['frequency'] = df['claim_amount'].apply(lambda x: 1 if x > 0 else 0)  # satın alma durumuna göre frequency değerini oluştur

# Filter out rows where vintage is 0
df = df[df['vintage'] != 0].copy()  # vintage değeri sıfırdan farklı olan satırları kullanıyoruz

# Ensure recency values are less than or equal to T
def calculate_recency(row):
    if row['frequency'] == 0:
        return 0  # Eğer frequency değeri sıfırsa, recency değeri sıfır olur
    else:
        return min(row['T'], (today_date.toordinal() - (today_date - dt.timedelta(weeks=row['vintage'])).day))

df['recency'] = df.apply(calculate_recency, axis=1)

# Adım 2: BetaGeoFitter modelini eğitin
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(df['frequency'], df['recency'], df['T'])

# Adım 3: CLTV tahmini ve sıralama
t = 1  # 3 aylık CLTV tahmini
df["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(t, df['frequency'], df['recency'], df['T'])

# Tahmin edilen satın alma sayısına göre sıralama
top_10_customers = df.sort_values(by="predicted_purchases", ascending=False)[:150]

# Adım 4: Sonuçları görüntüleme
print("3 Aylık Tahmini Satın Alma Sayısına Göre İlk 10 Müşteri:\n")
print(top_10_customers[['id', 'predicted_purchases']])

# Adım 5: CSV olarak dışa aktarma
top_10_customers.to_csv("top_10_customers_predicted_purchases.csv", index=False)

