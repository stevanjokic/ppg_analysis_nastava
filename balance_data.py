import pandas as pd
import numpy as np

inputFn = "ppg_template_hb_6k.csv"
outputFn = "ppg_template_hb_6k_bal.csv"

# Učitaj CSV fajl
df = pd.read_csv("ppg/" + inputFn)

# Filtriraj samo pulse u opsegu 50-85
df = df[(df["pulse"] >= 50) & (df["pulse"] <= 85) & (df["age"] >= 20) & (df["age"] <= 85)]
# df = df[ (df["age"] >= 20) & (df["age"] <= 85) ]


# Brojanje pojavljivanja svake starosne grupe
age_counts = df["age"].value_counts()

# Pronađi maksimalan broj uzoraka u nekoj starosnoj grupi
max_count = age_counts.max()

# Kreiraj novi dataframe sa ujednačenom raspodelom
balanced_df = pd.DataFrame()

for age, count in age_counts.items():
    # Filtriraj sve uzorke sa tom starosnom grupom
    subset = df[df["age"] == age]
    
    # Ako ih ima manje od max_count, nasumično ih ponovi
    if count < max_count:
        subset = subset.sample(n=max_count, replace=True, random_state=42)  # Oversampling
        
    # Dodaj u novi dataframe
    balanced_df = pd.concat([balanced_df, subset])

# Proveri novu raspodelu podataka
print(balanced_df["age"].value_counts())

# Snimi izbalansirani dataset
balanced_df.to_csv("ppg/" + outputFn, index=False)
