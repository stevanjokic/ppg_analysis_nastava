print("started")
from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

print("after imports")

# Load the model
# model = load_model('Epohe2000inputNurons512_outputcsv.h5')
# model = load_model('Epohe2000_two_hidden_outputcsv.h5')
# modelFn = "template_hb_sd_corr.h5"
modelFn = "ppg_tmp_170k_30.h5"
inputFn = "ppg_template_170k.csv"
outputFn = "result_age_predict_ann_template_hb_170k.txt"

model = load_model('model/' + modelFn)
inputFn = "ppg/ppg_template_170k.csv"

print("after load model, read data")


# input_data = pd.read_csv('data/' + inputFn)
input_data = pd.read_csv('ppg/' + inputFn)
input_data = input_data[(input_data["hr"] >= 50) & (input_data["hr"] <= 85) 
    & (input_data["age"] >25)  & (input_data["age"] < 85)]

# ppg_signals = input_data['data'].apply(eval).tolist() 
ppg_signals = input_data['ppg'].apply(eval).tolist() 
hr = input_data['hr'].tolist()
correct_ages = input_data['age'].tolist()
data_id = input_data['data_id'].tolist()

print("data read")
# ppg_signals = pad_sequences(ppg_signals, maxlen=2682, truncating='post', padding='post')

# Predict the age for each PPG signal
ages_predicted = model.predict(ppg_signals)

correct_ages = np.array(correct_ages) *100
ages_predicted = ages_predicted * 100

# Calculate error metrics
mae = mean_absolute_error(correct_ages, ages_predicted)
mse = mean_squared_error(correct_ages, ages_predicted)

print("writing file")
with open("model/" + outputFn, "w") as outFile:
    outFile.write("id,real age,predicted age,error,|error|\n")
    for i in range(len(correct_ages)):
        # print(f"id:{data_id[i]} Actual Age: {round(correct_ages[i])}, Predicted Age: {round(ages_predicted[i][0])}")
        outFile.write(f"{data_id[i]},{round(correct_ages[i])},{round(ages_predicted[i][0])},{round(ages_predicted[i][0]-correct_ages[i])},{abs(round((ages_predicted[i][0]-correct_ages[i])))}\n")

    outFile.write(f"\nmae:{round(mae)}\nmse:{round(mse)}")


print(f"Mean Absolute Error: {round(mae)}")
print(f"Mean Squared Error: {round(mse)}")

# If you also want to print each individual prediction alongside the actual age
# with open("model/result_age_predict_ann_template_hb_sd.txt", "w") as outFile:
#     outFile.write("id,real age,predicted age,|error|\n")
#     for i in range(len(correct_ages)):
#         print(f"id:{data_id[i]} Actual Age: {round(100*correct_ages[i])}, Predicted Age: {round(100*ages_predicted[i][0])}")
#         outFile.write(f"{data_id[i]},{round(100*correct_ages[i])},{round(100*ages_predicted[i][0])},{abs(round(100*(ages_predicted[i][0]-correct_ages[i])))}\n")

#     outFile.write(f"\nmae:{round(100*mae)}\nmse:{round(100*mse)}")

# print(f"Mean Absolute Error: {round(100*mae)}")
# print(f"Mean Squared Error: {round(100*mse)}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
# correct_ages = correct_ages*100
# correct_ages = np.array(correct_ages) *100
# ages_predicted = ages_predicted * 100

plt.scatter(correct_ages, ages_predicted, color='blue', alpha=0.7, edgecolors='black', label='Real vs predicted age')

# Dodavanje naslova i oznaka osa
plt.title("Age prediction")
plt.xlabel("Correct age")
plt.ylabel("Predicted age")
plt.ylim(10, 90)
plt.xlim(10, 90)

m, b = np.polyfit(correct_ages, ages_predicted, 1)  # 1 označava linearnu regresiju
m = m.item()
b = b.item()
# Generisanje X vrednosti za crtu regresione prave
x_range = np.linspace(min(correct_ages), max(correct_ages), 200)  # 200 tačaka između min i max X
y_pred = m * x_range + b  # Izračunavanje Y vrednosti

# Iscrtavanje regresione prave
plt.plot(x_range, y_pred, color='green', linewidth=2, label=f"Regression: y = {round(m, 3)}x + {round(b, 1)}")


plt.legend()
plt.grid(True)

# Prikazivanje grafa
plt.show()
