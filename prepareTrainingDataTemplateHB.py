import csv
import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks

startInd = 10
endInd = 80

outFileName = 'trainingDataTemplateHB.csv'

def normalize_ppg(ppg_signal):
    # Normalize all values to be between -1 and 1
    #return ppg_signal/np.max(np.abs(ppg_signal))
    return 2.*(ppg_signal - np.min(ppg_signal))/np.ptp(ppg_signal)-1

with open('ppg/ppg_sample_template_hb.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) # Skip the header row if present

    # Create an output CSV file for storing the wavelet coefficients
    with open('data/' + outFileName, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['data_id', 'age', 'data']) # Write header row
        for row in reader:
            data_id = row[0]
            age = float(row[1])/100.0
            ppg_data = np.array(eval(row[2])) # Convert the string to an array
            
            ppg_data = normalize_ppg(ppg_data[startInd:endInd+1])

            samples_str = ','.join(str(c) for c in ppg_data)
                        # Write the data to the output CSV file                    
            writer.writerow([data_id, age, samples_str])
                   

print("Training data saved to " + outFileName)