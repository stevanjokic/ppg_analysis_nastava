import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences

from keras.optimizers import Adam

inputFile = "trainingDataTemplateHB_SD_corr.csv"
outputFile= "template_hb_sd_corr.h5"
# Read the .csv file
# df = pd.read_csv('data/trainingDataTemplateHB_SD.csv')
df = pd.read_csv('data/' + inputFile)
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
model.add(Dense(128, input_dim=len(X[0]), activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mae', metrics=['mae'], optimizer=Adam(learning_rate=0.0003))

model.fit(X_train, y_train, epochs=15, batch_size=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print('Test loss:', loss)

# Save the model

model.save('model/' + outputFile)
print('saved model:' + outputFile)