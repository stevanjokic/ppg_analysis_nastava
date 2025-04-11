print("started")
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

print("after imports")

inputFn = "ppg_template_hb_6k_bal.csv"
modelFn = "ppg_tmp_hb_6k_bal_cnn1.h5"

ppgStart = 5
ppgEnd = 80

epoch = 10
batch_size = 8

def normalize_ppg(ppg_signal):
    ppg_signal = np.array(ppg_signal, dtype=np.float32)
    min_val, max_val = np.min(ppg_signal), np.max(ppg_signal)
    return 2 * (ppg_signal - min_val) / (max_val - min_val) - 1  # Normalizacija u opseg [-1, 1]

def normalize_0_1(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def sd(ppg_data):
    ppg_sd = np.zeros(len(ppg_data))
    for i in range(3, len(ppg_data)-3):
        ppg_sd[i] = -ppg_data[i-2] + 16*ppg_data[i-1] - 30*ppg_data[i] + 16*ppg_data[i+1] - ppg_data[i+2]
    return ppg_sd

print("loading data")

df = pd.read_csv("ppg/" + inputFn)
df = df[(df["pulse"] >= 50) & (df["pulse"] <= 85) & (df["age"] >= 20) & (df["age"] <= 85)]

# id,age,gender,pulse,data

ppg = df['data'].apply(lambda x: np.array([int(i) for i in x.split(',')])).values
ppg = np.array([normalize_0_1(x) for x in ppg])
ppg_sd = np.array([np.gradient(np.gradient(x)) for x in ppg])
# ppg_sd = np.array([sd(x) for x in ppg])
ppg_sd = np.array([normalize_0_1(x) for x in ppg_sd])

ppg_sub = np.array([x[ppgStart:ppgEnd] for x in ppg])
y_age = df['age'].values/100
gender = df['gender'].values
hr = df['pulse'].values

X_hr = df['pulse'].values.reshape(-1, 1)  # Puls
X_gender = df['gender'].values.reshape(-1, 1) 

# plt.plot(ppg[34], label='PPG niz 1')
# plt.plot(ppg_sd[34], label='PPG niz 1')
# plt.show()

# input_data = np.column_stack((hr, gender, ppg_sub))
# output_data = y_age

ppg_combined = np.stack((ppg, ppg_sd), axis=-1) 

print("data loaded")

input_ppg = Input(shape=(ppg_combined.shape[1], 2), name="ppg_input")  # 2 kanala (PPG + drugi izvod)
input_hr = Input(shape=(1,), name="hr_input")  #  (HR)
input_gender = Input(shape=(1,), name="gender_input")  # gender

x = Conv1D(filters=32, kernel_size=5, padding="same")(input_ppg)
x = LeakyReLU(alpha=0.01)(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(filters=64, kernel_size=3, padding="same")(x)
x = LeakyReLU(alpha=0.01)(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Flatten()(x)

# Kombinacija sa HR i Gender ulazima
merged = Concatenate()([x, input_hr, input_gender])

# Fully connected slojevi
dense = Dense(128, activation='relu')(merged)
dense = Dropout(0.2)(dense)
dense = Dense(64, activation='relu')(dense)
dense = Dropout(0.2)(dense)
dense = Dense(32, activation='relu')(dense)
output = Dense(1, activation='linear', name="age_output")(dense)  # Predikcija godina

# Kreiranje modela
model = Model(inputs=[input_ppg, input_hr, input_gender], outputs=output)

# Kompajliranje modela
optimizer = Adam(learning_rate=0.0005)
model.compile(loss="mae", optimizer=optimizer, metrics=["mae"])

print("starting training")
# Treniranje modela
history = model.fit(
    [ppg_combined, X_hr, X_gender], y_age,
    epochs=epoch,
    batch_size=batch_size,
    validation_split=0.2
)

# Evaluacija modela
loss, mae = model.evaluate([ppg_combined, X_hr, X_gender], y_age)
print(f"Final Loss: {loss}, MAE: {mae}")

# Snimanje modela
model.save("model/" + modelFn)
print("Model sačuvan kao " + modelFn)
