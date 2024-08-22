import pandas as pd
import numpy as np
import csv

displacement = []
force = []

file_path=r'C:/Users/PMLS/Desktop/Hassaan Digitrain/Hassaan Digitain/CA_MMB_M4_0,43.csv'
with open(file_path,'r') as file:
    reader = csv.reader(file, delimiter=';')
    next(reader)  # Skip header
    for row in reader:
                # Replace commas with periods
                row = [entry.replace(',', '.') for entry in row]
                if float(row[2]) >= 0:
                    displacement.append(float(row[2]))  # Column 3 for Displacement
                    force.append(float(row[1]))  # Column 2 for Force

data = {
    'Displacement': displacement,
    'Force': force
}


original_data=pd.DataFrame(data)

def generate_synthetic_data(data, num_samples=25000, variation_factor=0.1):

    synthetic_data = data.sample(n=num_samples, replace=False).reset_index(drop=True)
    noise = np.random.randint(10) * variation_factor
    synthetic_data += noise
    
    return synthetic_data

# Generate and save multiple synthetic CSV files
for i in range(15):
    synthetic_data = generate_synthetic_data(original_data, num_samples=25000, variation_factor=0.1)
    synthetic_data.to_csv(f'synthetic_data_{i+1}.csv', index=False)


print("Synthetic CSV files have been created.")
