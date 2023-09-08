from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Load the model
# model = load_model('Epohe2000inputNurons512_outputcsv.h5')
# model = load_model('Epohe2000_two_hidden_outputcsv.h5')
model = load_model('small.h5')

# Load the input CSV file
input_data = pd.read_csv('Data/trainingData.csv')

# Extract data from the DataFrame
ppg_signals = input_data['data'].apply(eval).tolist()  # Convert string to list
correct_ages = input_data['age'].tolist()

# If your model was trained on padded sequences,
# make sure to pad your input data in the same way
# ppg_signals = pad_sequences(ppg_signals, maxlen=2682, truncating='post', padding='post')

# Predict the age for each PPG signal
ages_predicted = model.predict(ppg_signals)

# Calculate error metrics
mae = mean_absolute_error(correct_ages, ages_predicted)
mse = mean_squared_error(correct_ages, ages_predicted)

print(f"Mean Absolute Error: {round(100*mae)}")
print(f"Mean Squared Error: {round(100*mse)}")

# If you also want to print each individual prediction alongside the actual age
for i in range(len(correct_ages)):
    print(f"Actual Age: {round(100*correct_ages[i])}, Predicted Age: {round(100*ages_predicted[i][0])}")
