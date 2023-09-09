import csv
import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks

leftL = 20
rightL = 50

# Normalizing
def normalize_ppg(ppg_signal):
    # Normalize all values to be between -1 and 1
    #return ppg_signal/np.max(np.abs(ppg_signal))
    return 2.*(ppg_signal - np.min(ppg_signal))/np.ptp(ppg_signal)-1

# Read the input CSV file
with open('data/sredjenExcel.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) # Skip the header row if present

    # Create an output CSV file for storing the wavelet coefficients
    with open('data/trainingDataSDPPG.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['data_id', 'age', 'data']) # Write header row
        for row in reader:
            data_id = row[0]
            age = float(row[1])/100.0
            ppg_data = np.array(eval(row[2])) # Convert the string to an array
            ppg_sd = np.zeros(len(ppg_data))
            
            for i in range(3, len(ppg_data)-3):
                # ppg_sd[i] = -ppg_data[i-2] + 16*ppg_data[i-1] - 30*ppg_data[i] + 16*ppg_data[i+1] - ppg_data[i+2]
                ppg_sd[i] = ppg_data[i-1] - 2*ppg_data[i] + ppg_data[i+1]
            
            peaks = signal.find_peaks(ppg_data, 50)[0]   
            
            for peak in peaks:
                if peak>2*leftL and peak<ppg_data.size-2*rightL:   

                    samples_ppg = normalize_ppg(np.array(ppg_data[peak-leftL:peak+rightL]))
                    samples = normalize_ppg(np.array(ppg_sd[peak-leftL:peak+rightL]))
                    
                    # samples = normalize_ppg(np.concatenate([left_samples, right_samples]))
                    if samples_ppg[0]<-.1: # verovatno greske u detekciji pikova
                        # Convert the coefficients to a string representation
                        samples_str = ','.join(str(c) for c in samples)
                        # Write the data to the output CSV file                    
                        writer.writerow([data_id, age, samples_str])

print("Training data saved to trainingDataSD.csv")