import pandas as pd
import matplotlib.pyplot as plt
from lifetimes.datasets import load_cdnow_summary
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_frequency_recency_matrix

# Adım 1: Veriyi yükleyin ve yapısına göz atın
data = load_cdnow_summary(index_col=[0])
print(data.head())

# Adım 2: BetaGeoFitter modelini eğitin ve özetini görüntüleyin
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(data['frequency'], data['recency'], data['T'])
print(bgf.summary)

# Adım 3: Frequency-Recency matrisini görselleştirme
plot_frequency_recency_matrix(bgf)
plt.show()

# Adım 4: Gelecekteki satın almaları tahmin etme
t = 1  # 1 dönemlik tahmin
data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, data['frequency'], data['recency'], data['T'])

# Tahmin edilen satın almalara göre veriyi sıralama ve en fazla satın alma tahmini olan 5 müşteri
predicted = data.sort_values(by='predicted_purchases',ascending = False).tail(5)
print(predicted)

# Tahmin edilen satın almaları görselleştirme
plt.figure(figsize=(10, 6))
plt.bar(predicted.index, predicted['predicted_purchases'], color='skyblue')
plt.xlabel('Customer ID')
plt.ylabel('Predicted Purchases')
plt.title('Top 5 Customers with Highest Predicted Purchases')
plt.show()
