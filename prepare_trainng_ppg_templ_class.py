import pandas as pd
import numpy as np

# from keras.models import Sequential
# from keras.layers import Dense

inputFn = "ppg_template_class_200.csv"
outputFn = 'training_ppg_template_class_200.csv'

def normalize_array(data):
    min_val = min(data)  # Minimalna vrednost
    max_val = max(data)  # Maksimalna vrednost
    return [(x - min_val) / (max_val - min_val) for x in data]

df = pd.read_csv('ppg/' + inputFn)

# print(df["template_ppg"])
print (df.columns)
print(df["age"][0])
print(df["age"][1])

df['class1'] = None
df['class2'] = None
df['class3'] = None
df['class4'] = None

for index, row in df.iterrows():
    ppg = list( map(int, df['template_ppg'][index].split(',')) )    
    ppg = normalize_array(ppg)
    # df.drop('template_ppg', axis=1)
    # df.loc[index, "template_ppg"] = ppg
    df['template_ppg'][index] = ppg
    ppgClass = df['class'][index]
    if (ppgClass==1):
        df.loc[index, 'class1'] = 1
        df.loc[index, 'class2'] = 0
        df.loc[index, 'class3'] = 0
        df.loc[index, 'class4'] = 0
    
    elif (ppgClass==2):
        df.loc[index, 'class1'] = 0
        df.loc[index, 'class2'] = 1
        df.loc[index, 'class3'] = 0
        df.loc[index, 'class4'] = 0
    elif (ppgClass==3):
        df.loc[index, 'class1'] = 0
        df.loc[index, 'class2'] = 0
        df.loc[index, 'class3'] = 1
        df.loc[index, 'class4'] = 0
    elif (ppgClass==4):
        df.loc[index, 'class1'] = 0
        df.loc[index, 'class2'] = 0
        df.loc[index, 'class3'] = 0
        df.loc[index, 'class4'] = 1    
    
df.to_csv('data/' + outputFn, index=False)
