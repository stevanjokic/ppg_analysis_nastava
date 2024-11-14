import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences


# Read the .csv file
df = pd.read_csv('data/trainingDataTemplateHB.csv')
# df = pd.read_csv('data/trainingDataSDPPG.csv')
# df = pd.read_csv('data/trainingDataSDPPG.csv')
# df = pd.read_csv('data/trainingData_DB6.csv')


# Convert 'Coefficients' from string to list using ast.literal_eval
df['data'] = df['data'].apply(lambda x: ast.literal_eval(x))

# Padding
X = pad_sequences(df['data'].values, padding='post')


y = df['age'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Model architecture remains the same
model = Sequential()
model.add(Dense(100, input_dim=len(X[0]), activation='tanh'))
model.add(Dense(70, activation='tanh'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=100, batch_size=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print('Test loss:', loss)

# Save the model
model.save('model/template_hb.h5')
