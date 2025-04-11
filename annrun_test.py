from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

modelFn = "template_hb.h5"
inputFn = "trainingDataTemplateHB.csv"
outputFn = "result_age_predict_ann_template_hb.txt"
# Load the model
# model = load_model('Epohe2000inputNurons512_outputcsv.h5')
# model = load_model('Epohe2000_two_hidden_outputcsv.h5')
modelFn = load_model('model/' + modelFn)

# Load the input CSV file
input_data = pd.read_csv('data/' + inputFn)

# Extract data from the DataFrame
ppg_signals = input_data['data'].apply(eval).tolist()  # Convert string to list
correct_ages = input_data['age'].tolist()
data_id = input_data['data_id'].tolist()

# If your model was trained on padded sequences,
# make sure to pad your input data in the same way
# ppg_signals = pad_sequences(ppg_signals, maxlen=2682, truncating='post', padding='post')

# Predict the age for each PPG signal
ages_predicted = modelFn.predict(ppg_signals)

# Calculate error metrics
mae = mean_absolute_error(correct_ages, ages_predicted)
mse = mean_squared_error(correct_ages, ages_predicted)

# If you also want to print each individual prediction alongside the actual age
with open("model/" + outputFn, "w") as outFile:
    outFile.write("id,real age,predicted age,|error|\n")
    for i in range(len(correct_ages)):
        print(f"id:{data_id[i]} Actual Age: {round(100*correct_ages[i])}, Predicted Age: {round(100*ages_predicted[i][0])}")
        outFile.write(f"{data_id[i]},{round(100*correct_ages[i])},{round(100*ages_predicted[i][0])},{abs(round(100*(ages_predicted[i][0]-correct_ages[i])))}\n")

    outFile.write(f"\nmae:{round(100*mae)}\nmse:{round(100*mse)}")

print(f"Mean Absolute Error: {round(100*mae)}")
print(f"Mean Squared Error: {round(100*mse)}")
