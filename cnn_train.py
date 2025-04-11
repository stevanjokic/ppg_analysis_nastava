print("started")
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
# from imblearn.over_sampling import RandomOverSampler # problemi sa bibliotekom
from scipy.interpolate import interp1d

print("after imports")

inputFn = "balanced_ppg_template.csv"
modelFn = "ppg_tmp_170k_bal_mae.h5"
epoch = 15

def normalize_ppg(ppg_signal):
    ppg_signal = np.array(ppg_signal, dtype=np.float32)
    min_val, max_val = np.min(ppg_signal), np.max(ppg_signal)
    return 2 * (ppg_signal - min_val) / (max_val - min_val) - 1  # Normalizacija u opseg [-1, 1]


df = pd.read_csv("ppg/" + inputFn)
# Filtriranje podataka: koristimo samo redove gde je HR (puls) u opsegu 50-85
df = df[(df["hr"] >= 50) & (df["hr"] <= 85) & (df["age"] >= 20) & (df["age"] <= 85)]

# Parsiranje PPG signala
df['ppg'] = df['ppg'].apply(lambda x: np.array([float(i) for i in x.strip('"').split(',')]))  # Pretvaranje u niz brojeva
df['ppg'] = df['ppg'].apply(normalize_ppg)  # Normalizacija

# Ekstrahovanje ulaznih podataka
X_ppg = np.stack(df['ppg'].values)  # Pretvaranje liste nizova u numpy array
X_hr = df['hr'].values.reshape(-1, 1)  # Heart rate kao dodatni feature
y_age = df['age'].values /100  # Izlaz (godine)

# Oblikovanje podataka
X_ppg = X_ppg.reshape(X_ppg.shape[0], X_ppg.shape[1], 1)  # (broj uzoraka, dužina signala, 1 kanal)

# **Kreiranje modela sa 2 ulaza (PPG + dodatni ulaz hr)**
input_ppg = Input(shape=(X_ppg.shape[1], 1), name="ppg_input")
input_hr = Input(shape=(1,), name="hr_input")

# CNN slojevi za PPG
x = Conv1D(filters=64, kernel_size=5, padding="same")(input_ppg)
x = LeakyReLU(alpha=0.01)(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(filters=128, kernel_size=3, padding="same")(x)
x = LeakyReLU(alpha=0.01)(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Flatten()(x)

merged = tf.keras.layers.concatenate([x, input_hr])

# Dense slojevi
dense = Dense(256, activation='relu')(merged)
dense = Dropout(0.2)(dense)
dense = Dense(128, activation='relu')(dense)
dense = Dropout(0.2)(dense)
dense = Dense(64, activation='relu')(dense)
output = Dense(1, activation='linear', name="age_output")(dense)  # Izlaz je predikcija godina


# Kreiranje modela
model = Model(inputs=[input_ppg, input_hr], outputs=output)

# Kompajliranje modela
optimizer = Adam(learning_rate=0.0002)
model.compile(optimizer = optimizer, loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Treniranje modela
history = model.fit(
    [X_ppg, X_hr], y_age,
    epochs=epoch,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluacija modela
loss, mae = model.evaluate([X_ppg, X_hr], y_age)
print(f"Final Loss: {loss}, MAE: {mae}")

# **Snimanje modela**
model.save("model/" + modelFn)
print("Model sačuvan kao " + modelFn)
