import pandas as pd
import numpy as np
from openpyxl import load_workbook

# Constants
B = 20.075   # Width in mm
h = 4.281    # Thickness in mm
a = 22       # Crack length in mm
E11 = 171420  # Longitudinal modulus of elasticity in MPa (N/mm^2)
G13 = 4480    # Shear modulus out of plane in MPa (N/mm^2)
support_length = 50  # Example value for support length in mm

# Read the Excel sheet with G1, G2
file_path = r"D:\Aditechsim\DigiTain\data_for_g1\Extrapolated Data\synthetic_data (10).xlsx"
df = pd.read_excel(file_path)

# Assuming the Excel sheet contains columns for 'G1' and 'G2' in N·mm (Energy release rates)
G1 = df['G1'].values # Already in N·mm
G2 = df['G2'].values # Already in N·mm

# Initialize empty lists to store calculated values
force_list = []
displacement_list = []
bending_stress_list = []

# Loop through each row and calculate force, displacement, and bending stress
for i in range(len(G1)):
    # Calculate forces using the correct formulas
    force_bending = G1[i] / support_length  # Force due to G1 (Mode I) in N
    force_shear = G2[i] / support_length     # Force due to G2 (Mode II) in N
    
    # Total force is the combination of both bending and shear forces
    total_force = np.sqrt(force_bending**2 + force_shear**2)  # Resultant force in Newtons
    
        # Calculate moment of inertia (in mm^4)
    moment_of_inertia = (B * h**3) / 12  # B: width in mm, h: thickness in mm

    # Calculate stiffness (in N/mm) using E11 in N/mm^2
    stiffness = (3 * E11 * moment_of_inertia) / (support_length**3)  # support length is already in mm

    # Calculate displacement based on total force (in mm)
    displacement = total_force / stiffness  # Displacement in mm

    
    # Calculate bending stress using span length (support length constant)
    bending_stress = (total_force * support_length) / (4 * B * h**2)  # Bending stress in MPa
    
    # Append calculated values to the lists
    force_list.append(total_force)
    displacement_list.append(displacement)
    bending_stress_list.append(bending_stress)

# Add the calculated columns to the DataFrame
df['Force_N'] = force_list
df['Displacement_mm'] = displacement_list
df['Bending_Stress_MPa'] = bending_stress_list

# Save to a new Excel file
new_file_path = r"D:\Aditechsim\DigiTain\data_for_g1\Extrapolated Data\updated_synthetic_data_FDchange.xlsx"
with pd.ExcelWriter(new_file_path, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Sheet1')

print(f"Updated file saved to {new_file_path}")
