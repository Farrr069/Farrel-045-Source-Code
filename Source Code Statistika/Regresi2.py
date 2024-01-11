import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Baca data dari file CSV (gantilah 'nama_file.csv' dengan nama file yang sesuai)
data = pd.read_csv(r'C:\Statistika\codingan\heart.csv', delimiter=",")

# Tentukan variabel independen dan dependen
X = data[['sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'target']]  # Ganti dengan variabel independen yang sesuai
Y = data['age']  # Ganti dengan variabel dependen yang sesuai

# Bagi data menjadi data latih dan data uji
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Inisialisasi model regresi linier
model = LinearRegression()

# Latih model menggunakan data latih
model.fit(X_train, Y_train)

# Lakukan prediksi menggunakan data uji
Y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Tampilkan hasil evaluasi
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Visualisasi hasil prediksi
plt.scatter(Y_test, Y_pred)
plt.xlabel('Nilai Sebenarnya')
plt.ylabel('Prediksi')
plt.title('Hasil Prediksi vs. Nilai Sebenarnya')
plt.show()

# Analisis lebih lanjut dengan Statsmodels
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(Y_train, X_train_sm).fit()

# Tampilkan ringkasan model
print("\nRingkasan Model Regresi (Statsmodels):")
print(model_sm.summary())
