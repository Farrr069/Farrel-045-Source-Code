import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca data dari file CSV
data = pd.read_csv(r'C:\Statistika\codingan\heart.csv', delimiter=',')

# Mengonversi 'Level' ke variabel dummy
data = pd.get_dummies(data, columns=['sex'], drop_first=True)

# Mengidentifikasi kolom numerik dan kolom kategorikal yang akan diabaikan
kolom_numerik = data.select_dtypes(include=['float64', 'int64']).columns
kolom_abaikan = ['target']  # Ganti dengan nama kolom non-numerik yang ingin diabaikan

# Mengabaikan kolom non-numerik
kolom_analisis = [kolom for kolom in kolom_numerik if kolom not in kolom_abaikan]

# Menentukan variabel dependen dan independen
X = data[kolom_analisis]
y = data['age']  # Ganti dengan nama kolom dependen yang sesuai

# Membuat model regresi
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Mendapatkan residual dari model
residuals = model.resid

# Uji Normalitas (Shapiro-Wilk)
stat, p_value = shapiro(residuals)
print("Shapiro-Wilk Test Statistic:", stat)
print("P-value:", p_value)

# Visualisasi QQ Plot
sm.qqplot(residuals, line='s')
plt.show()

# Uji Heteroskedastisitas (Breusch-Pagan Test)
lm, lm_p_value, fval, fval_p_value = het_breuschpagan(residuals, X)
print("Lagrange Multiplier (LM) Test Statistic:", lm)
print("LM Test P-value:", lm_p_value)
print("F-statistic:", fval)
print("F-statistic P-value:", fval_p_value)

# Visualisasi Scatterplot Residual vs Fitted
fitted_values = model.fittedvalues
sns.scatterplot(x=fitted_values, y=residuals)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

# Uji Autokorelasi (Durbin-Watson Test)
dw_statistic = durbin_watson(residuals)
print("Durbin-Watson Statistic:", dw_statistic)

# Visualisasi Correlogram
plot_acf(residuals, lags=40)
plt.show()

# Hasil ringkasan model
print("\nRingkasan Model Regresi:")
print(model.summary())
