import csv
import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks
from pywt import wavedec

leftL = 30
rightL = 60

# Read the input CSV file
with open('data/sredjenExcel.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) # Skip the header row if present

    # Create an output CSV file for storing the wavelet coefficients
    with open('data/trainingData_DB6.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['id', 'age', 'data']) # Write header row
        for row in reader:
            data_id = row[0]
            age = float(row[1])/100.0
            ppg_data = np.array(eval(row[2])) # Convert the string to an array
            
            peaks = signal.find_peaks(ppg_data, distance=50)[0]

            for peak in peaks:
                if peak>2*leftL and peak<ppg_data.size-2*rightL:                    
                    samples = np.array(ppg_data[peak-leftL:peak+rightL])
                    coeffs = wavedec(samples, wavelet='db6')
                    
                    # Convert the coefficients to a string representation
                    samples_str = ','.join(str(c) for c in coeffs[0])
                    # Write the data to the output CSV file                    
                    writer.writerow([data_id, age, samples_str])

print("Training data saved")