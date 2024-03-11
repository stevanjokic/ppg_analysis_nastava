import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import pad_sequences

# Usage Long Short-Term Memory networks

# Read the .csv file
df = pd.read_csv('data/trainingData_DB6.csv')


# Convert 'Coefficients' from string to list using ast.literal_eval
df['data'] = df['data'].apply(lambda x: ast.literal_eval(x))

# Padding
X = pad_sequences(df['data'].values, padding='post')


y = df['age'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('len(X[0]):', len(X[0]))
# Model architecture remains the same
model = Sequential()

# Add the input layer
#model.add(Dense(64, input_dim=len(X[0]), activation='sigmoid'))
model.add(LSTM(64, activation='sigmoid', input_shape=(len(X[0]), 1),return_sequences=True))
# Add the first hidden layer
#model.add(Dense(32, activation='sigmoid'))
model.add(LSTM(64, return_sequences=True))

# Add the second hidden layer
#model.add(Dense(16, activation='sigmoid'))
model.add(LSTM(64, return_sequences=True))

# Add the third hidden layer
#model.add(Dense(8, activation='sigmoid'))
model.add(LSTM(64))

# Add the output layer
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=100, batch_size=8)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print('Test loss:', loss)

# Save the model
model.save('model/small_LSTM_3_64_PPG.h5')
