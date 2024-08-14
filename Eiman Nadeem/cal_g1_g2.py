import pandas as pd
import numpy as np

# Load the CSV file with the correct delimiter
csv_path = 'D:\\Aditechsim\\DigiTain\\CA_MMB_M2_0,77.csv'
data = pd.read_csv(csv_path, delimiter=',')  # Update delimiter to comma

# Check if data is loaded correctly or if everything is in one column
print("Original Columns:", data.columns)
print(data.head())

# If everything is in one column, manually split the columns
if len(data.columns) == 1:
    data = data.iloc[:, 0].str.split(',', expand=True)  # Use ',' for splitting
    # Rename the columns manually
    data.columns = ['time', 'force', 'normal_displacement', 'shear_displacement', 'Deflection', 'Bending_stress']

# Print the corrected column names
print("Corrected Columns:", data.columns)

# Clean the data: Remove leading zeros and semicolons, then convert to numeric
def clean_column(column):
    column = column.replace({';': '', ' ': '', ',': '.'}, regex=True)
    return pd.to_numeric(column, errors='coerce')

# Apply the cleaning function to the relevant columns
data['force'] = clean_column(data['force'])
data['normal_displacement'] = clean_column(data['normal_displacement'])
data['shear_displacement'] = clean_column(data['shear_displacement'])
data['Deflection'] = clean_column(data['Deflection'])
data['Bending_stress'] = clean_column(data['Bending_stress'])

# Convert displacements from millimeters to meters
data['normal_displacement'] /= 1000
data['shear_displacement'] /= 1000

# Check for missing values after cleaning
print("Missing values in each column:")
print(data.isna().sum())

# Fill or handle missing values if necessary
data = data.fillna(method='ffill').fillna(method='bfill')  # Forward and backward fill

# Ensure data is correctly aligned and no NaNs
print("Cleaned Data:")
print(data.head())

# Now the columns should be numeric, and you can perform calculations
force = data['force'].values  
normal_displacement = data['normal_displacement'].values  
shear_displacement = data['shear_displacement'].values  

# Ensure lengths match
assert len(force) == len(normal_displacement) == len(shear_displacement), "Mismatch in data lengths"

# Calculate the average force for trapezoidal integration
force_avg = (force[:-1] + force[1:]) / 2

# Calculate differences in displacements for trapezoidal integration
delta_normal = np.diff(normal_displacement)
delta_shear = np.diff(shear_displacement)

# Check that the lengths of arrays match for integration
assert len(force_avg) == len(delta_normal) == len(delta_shear), "Mismatch in data lengths for integration"

# Use the trapezoidal rule to calculate G1 and G2
G1 = np.sum(force_avg * delta_normal)
G2 = np.sum(force_avg * delta_shear)

# Adjust by the specimen width B (use the actual width)
B = 20.0375  # Adjust this value as needed
G1 /= B
G2 /= B

# Output the calculated G1 and G2
print(f'G1: {G1}')
print(f'G2: {G2}')
