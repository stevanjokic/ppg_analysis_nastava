import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt

# Učitavanje modela
modelFn = "ppg_tmp_hb_6k_bal_cnn1.h5"
model = load_model("model/" + modelFn)

# Učitavanje podataka
inputFn = "ppg_template_hb_6k_bal.csv"
df = pd.read_csv("ppg/" + inputFn)
df = df[(df["pulse"] >= 50) & (df["pulse"] <= 85) & (df["age"] >= 20) & (df["age"] <= 85)]

# Priprema podataka
def normalize_0_1(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

ppg = df['data'].apply(lambda x: np.array([int(i) for i in x.split(',')])).values
ppg = np.array([normalize_0_1(x) for x in ppg])
ppg_sd = np.array([np.gradient(np.gradient(x)) for x in ppg])
ppg_sd = np.array([normalize_0_1(x) for x in ppg_sd])

ppg_combined = np.stack((ppg, ppg_sd), axis=-1) 
y_age = df['age'].values/100
X_hr = df['pulse'].values.reshape(-1, 1)
X_gender = df['gender'].values.reshape(-1, 1) 

# Selektovanje 20% slučajnih podataka
np.random.seed(42)
indices = np.random.choice(len(ppg_combined), size=int(len(ppg_combined) * 0.2), replace=False)

# ppg_sample = ppg_combined[indices]
# hr_sample = X_hr[indices]
# gender_sample = X_gender[indices]
# age_real = y_age[indices]
ppg_sample = ppg_combined
hr_sample = X_hr
gender_sample = X_gender
age_real = y_age

# Predikcija starosti koristeći model
age_pred = model.predict([ppg_sample, hr_sample, gender_sample])

age_pred = age_pred * 100
age_real = age_real * 100

data = {
    'data_id': df['data_id'], 
    'real_age': age_real,
    'predicted_age': np.squeeze(age_pred),    
    'dif': np.squeeze(age_pred) - np.squeeze(age_real)
}

dfCr = pd.DataFrame(data)

# Snimanje DataFrame-a u CSV fajl
dfCr.to_csv('model/res_' + inputFn, index=False)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
correlation_matrix = np.corrcoef(np.squeeze(age_real), np.squeeze(age_pred))
correlation_coefficient = correlation_matrix[0, 1]
print("Pearsonov koeficijent korelacije (R):", round(correlation_coefficient), 2)

r2 = round(r2_score(age_real, age_pred), 2)
print("Koeficijent determinacije (R^2):", r2)

mse = round(mean_squared_error(age_real, age_pred), 2)
rmse = round(np.sqrt(mse), 2)
mae = round(mean_absolute_error(age_real, age_pred), 2)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Squared Error (MSE):", mse)


# Crtanje scatter plot stvarne i predviđene starosti
plt.figure(figsize=(10, 6))
plt.plot([10, 90], [10, 90], 'r--')  # Dijagonala za idealne predikcije
plt.scatter(age_real, age_pred, alpha=0.02)

plt.xlabel('Real Age')
plt.ylabel('Predicted Age')
plt.title('Real vs Predicted Age')
plt.show()
