import numpy as np

print("This line will be printed.")



def normalize_ppg(ppg_signal):
    # Normalize all values to be between -1 and 1
    return 2.*(a1D - np.min(a1D))/np.ptp(a1D)-1

a1D = np.array([-20, -1, 0, 1, 2, 3])

#d = 2.*(a1D - np.min(a1D))/np.ptp(a1D)-1

print((a1D))
print(normalize_ppg(a1D))