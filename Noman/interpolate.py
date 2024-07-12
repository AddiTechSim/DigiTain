"""This program was written to understadn, how to do interpolation of a complex data. i.e in GA.csv
Initially, instead of doing "data = pd.read_csv(GA.csv)" I have define the raw data 
as a dictionary beacuse reading the csv file directly gives me conversion error.
To fix this, I am planning to use 'ast module'.
But just to prove the concept of interpolation on this complex data. 
This code work fine.
Next, will be finding ways to solve the data conversion issue. So, we don't have to 
define large dictionary for interpolation. Just use 'read_csv()' function, and the
program outputs the interpolated data

Method of interpolation: CUBIC SUPLINE
Why?: Because of it's smoothness & computational efficiency with large datasets

"""



import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Define the raw data as a dictionary for easier conversion to a DataFrame
data = {
    "Carbon-fabric_Training_TENSILE_FIBER1": [
        [[-1.0, -20000000000.0], [0.0, 0.0], [1.0, 20000000000.0]],
        [[-1.0, -20000000000.0], [0.0, 0.0], [1.0, 20000000000.0]],
        [[-1.0, -20000000000.0], [0.0, 0.0], [1.0, 20000000000.0]],
        [[-1.0, -20000000000.0], [1.0, 20000000000.0]],
        [[-1.0, -20000000000.0], [1.0, 20000000000.0]],
        [[-1.0, -20000000000.0], [1.0, 20000000000.0]],
        [[-1.0, -20000000000.0], [1.0, 20000000000.0]]
    ],
    "Carbon-fabric_Training_PLANE_SHEAR": [
        [[-1.5, -250000.0], [-0.5, -12500.0], [0.0, 0.0], [0.5, 12500.0], [1.5, 250000.0]],
        [[-1.5, -250000.0], [0.0, 0.0], [1.5, 250000.0]],
        [[-1.5, -250000.0], [0.0, 0.0], [1.5, 250000.0]],
        [[-1.5, -250000.0], [-0.5, -12500.0], [0.0, 0.0], [0.5, 12500.0], [1.5, 250000.0]],
        [[-1.5, -250000.0], [-0.5, -12500.0], [0.0, 0.0], [0.5, 12500.0], [1.5, 250000.0]],
        [[-1.5, -250000.0], [0.0, 0.0], [1.5, 250000.0]],
        [[-1.5, -250000.0], [0.0, 0.0], [1.5, 250000.0]]
    ],
    "Carbon-fabric_Training_TENSILE_FIBER2": [
        -25000000000, -20000000000, -19000000000, 0, 19000000000, 20000000000, 25000000000
    ],
    "max_# Translational_Displacement_3": [
        0.011718221, 0.006061279, 0.006051063, 0.00850612, 0.007420537, 0.007157162, 0.006120917
    ],
    "min_# Fiber_Angle_3": [
        58.20797348, 71.44635773, 71.23151398, 66.81512451, 66.76634979, 65.77740479, 70.83242035
    ],
    "min_# Shear_Angle_3": [
        -22.65366745, -24.08789253, -16.51135445, -18.23303986, -18.40634155, -17.8915844, -20.66315269
    ],
    "mean_# Shear_Stress_3": [
        -76.02532285, 138.9650966, 108.4289887, -547.6365809, 114.8558841, 165.8498476, -415.8896439
    ]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Preprocess the data to extract x and y values for interpolation
def flatten_pairs(pairs):
    # Flatten nested lists of pairs into a single list
    return [item for sublist in pairs for item in sublist]

# Flatten the 'Carbon-fabric_Training_TENSILE_FIBER1' column
tensile_fiber1_flat = df["Carbon-fabric_Training_TENSILE_FIBER1"].apply(flatten_pairs).tolist()
# Extract x and y values from the flattened data
tensile_fiber1_x = np.array([x for sublist in tensile_fiber1_flat for x in sublist[0::2]])
tensile_fiber1_y = np.array([y for sublist in tensile_fiber1_flat for y in sublist[1::2]])

# Sort the x and y values to ensure x is strictly increasing
sorted_indices = np.argsort(tensile_fiber1_x)
tensile_fiber1_x = tensile_fiber1_x[sorted_indices]
tensile_fiber1_y = tensile_fiber1_y[sorted_indices]

# Remove duplicates from the x-values and average corresponding y-values
unique_x, indices = np.unique(tensile_fiber1_x, return_index=True)
tensile_fiber1_y = tensile_fiber1_y[indices]

# Apply cubic spline interpolation on 'Carbon-fabric_Training_TENSILE_FIBER1'
cubic_spline_interpolator = CubicSpline(unique_x, tensile_fiber1_y)

# Generate interpolated values
x_new = np.linspace(min(unique_x), max(unique_x), 100)
y_new = cubic_spline_interpolator(x_new)

# Plot the results
plt.plot(unique_x, tensile_fiber1_y, 'o', label='Original data')
plt.plot(x_new, y_new, '-', label='Cubic spline interpolation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Cubic Spline Interpolation on Carbon-fabric_Training_TENSILE_FIBER1')
plt.show()
