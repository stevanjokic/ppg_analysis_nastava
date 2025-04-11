import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 📌 Naziv modela i CSV fajla
# modelFn = "model/ppg_tmp_170k_30.h5"
# inputFn = "ppg/ppg_template_170k.csv"
# outputFn = "model/ppg_predictions.csv"

modelFn = "model/ppg_tmp_170k_bal_mae.h5"
inputFn = "ppg/balanced_ppg_template.csv"
outputFn = "model/ppg_predictions_balanced.csv"


# 📌 Funkcija za normalizaciju PPG signala u opseg [-1, 1]
def normalize_ppg(ppg_signal):
    ppg_signal = np.array(ppg_signal, dtype=np.float32)
    min_val, max_val = np.min(ppg_signal), np.max(ppg_signal)
    return 2 * (ppg_signal - min_val) / (max_val - min_val) - 1  

# 📌 Učitavanje podataka iz CSV-a
df = pd.read_csv(inputFn)

# 📌 Filtriranje podataka: koristimo samo redove gde je HR (puls) u opsegu 50-85
df = df[(df["hr"] >= 50) & (df["hr"] <= 85) & (df["age"] >= 25) & (df["age"] <= 85) ]

# 📌 Parsiranje i normalizacija PPG signala
df['ppg'] = df['ppg'].apply(lambda x: np.array([float(i) for i in x.strip('"').split(',')]))  # Pretvaranje u niz brojeva
df['ppg'] = df['ppg'].apply(normalize_ppg)  # Normalizacija

# 📌 Priprema ulaznih podataka
X_ppg = np.stack(df['ppg'].values)  # Pretvaranje liste nizova u numpy array
X_hr = df['hr'].values.reshape(-1, 1)  # HR kao dodatni feature

# 📌 Oblikovanje podataka za predikciju
X_ppg = X_ppg.reshape(X_ppg.shape[0], X_ppg.shape[1], 1)  # (broj uzoraka, dužina signala, 1 kanal)

# 📌 Učitavanje modela
model = load_model(modelFn)

# 📌 Predikcija modela
predictions = model.predict([X_ppg, X_hr])

# 📌 Pretvaranje predikcija nazad u godine (×100 jer smo ih pri treniranju delili sa 100)
predicted_ages = predictions.flatten() * 100

# 📌 Računanje greške
actual_ages = df["age"].values  # Stvarne godine
prediction_errors = np.abs(actual_ages - predicted_ages)  # Apsolutna greška

# 📌 Računanje MSE i MAE
mse = mean_squared_error(actual_ages, predicted_ages)
mae = mean_absolute_error(actual_ages, predicted_ages)

print(f"📉 MSE (Srednja kvadratna greška): {mse:.2f}")
print(f"📊 MAE (Srednja apsolutna greška): {mae:.2f}")

# 📌 Dodavanje predikcija i greške u DataFrame
df["predicted_age"] = predicted_ages
df["prediction_error"] = prediction_errors  # Apsolutna greška

# 📌 Čuvanje rezultata u novi CSV fajl
df.to_csv(outputFn, index=False)
print(f"✅ Predikcije sačuvane u {outputFn}")

# 📌 Scatter graf: Stvarne vs Predikovane godine
plt.figure(figsize=(8, 6))
plt.scatter(actual_ages, predicted_ages, alpha=0.5, color='blue', label="Age prediction")
# plt.plot([min(actual_ages), max(actual_ages)], [min(actual_ages), max(actual_ages)], color='red', linestyle='--', label="Idealna linija (y=x)")
plt.xlabel("Real age")
plt.ylabel("Predicted age")
plt.ylim(10, 90)
plt.xlim(10, 90)
plt.title("Real vs Predicted age")
plt.legend()
plt.grid()
plt.show()
