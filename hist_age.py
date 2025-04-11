import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

inputFn = "ppg_template_170k.csv"
inputFn = "balanced_ppg_template.csv"

df = pd.read_csv("ppg/" + inputFn)

df = df[(df["hr"] >= 50) & (df["hr"] <= 85) & (df["age"] > 15) & (df["age"] <= 99) ]

plt.hist(df["age"], bins=5)

plt.title('Ages distribution')
plt.xlabel('Age')
plt.ylabel('Number of records')

plt.show()