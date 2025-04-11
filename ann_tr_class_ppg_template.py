import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

inputFn = "training_ppg_template_class_200.csv"

file_path = "data/" + inputFn  # Zamenite sa stvarnom putanjom
df = pd.read_csv(file_path)

# 2. Obrada podataka
# Pretvaranje `template_ppg` niza iz stringa u listu float vrednosti
df['template_ppg'] = df['template_ppg'].apply(lambda x: np.array(eval(x)))

# Kreiranje ulaznog niza (X) i izlaznih klasa (y)
X = np.stack(df['template_ppg'].values)  # Ulazne vrednosti (template_ppg nizovi)
y = df[['class1', 'class2', 'class3', 'class4']].values  # Izlazne klase

# 3. Deljenje na trenirajući i testirajući skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 4. Definisanje neuronske mreže
model = Sequential([
    Dense(256, activation='sigmoid', input_shape=(X_train.shape[1],)),  # Prvi sloj
    Dense(128, activation='sigmoid'),  # Skriveni sloj
    Dense(64, activation='sigmoid'),  # Skriveni sloj
    Dense(4, activation='sigmoid')  # Izlazni sloj sa 4 klase
])

# Kompajliranje modela
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Obuka modela
model.fit(X_train, y_train, epochs=40, batch_size=1, validation_split=0.1)

# 6. Evaluacija na test podacima
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test tačnost: {accuracy:.2f}")
