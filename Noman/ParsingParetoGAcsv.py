# -*- coding: utf-8 -*-

"""This program draw 4 plots of 4 output variables(Transational_Displacement etc) of GA.csv
   First I did parsing on Input variables i.e  TENSILE_FIBER1 && PLANE_SHEAR
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import ast

# Load the data
df = pd.read_csv('GA.csv')

# Function to parse the complex nested structures
def parse_complex_column(column):
    return column.apply(lambda x: np.array(ast.literal_eval(x)))

# Parse the complex columns
df['Carbon-fabric_Training_TENSILE_FIBER1'] = parse_complex_column(df['Carbon-fabric_Training_TENSILE_FIBER1'])
df['Carbon-fabric_Training_PLANE_SHEAR'] = parse_complex_column(df['Carbon-fabric_Training_PLANE_SHEAR'])

# Extract the independent variables
df['TENSILE_FIBER1_mean'] = df['Carbon-fabric_Training_TENSILE_FIBER1'].apply(lambda x: np.mean([val[1] for val in x]))
df['PLANE_SHEAR_mean'] = df['Carbon-fabric_Training_PLANE_SHEAR'].apply(lambda x: np.mean([val[1] for val in x]))

# Prepare the dataframe for regression
inputs = df[['TENSILE_FIBER1_mean', 'PLANE_SHEAR_mean']]
targets = df[['max_# Translational_Displacement_3', 'min_# Fiber_Angle_3', 'min_# Shear_Angle_3', 'mean_# Shear_Stress_3']]

# Dictionary to store the coefficients
coeff_dict = {}

# Perform linear regression and store the coefficients
for target in targets.columns:
    regr = linear_model.LinearRegression()
    regr.fit(inputs, df[target])
    coeff_dict[target] = regr.coef_

# Function to plot Pareto chart
def plot_pareto(coeff_dict, inputs):
    for target, coeffs in coeff_dict.items():
        coeff = [abs(i) for i in coeffs]
        c_nor = []
        c_abs = []
        for i in range(len(coeffs)):
            if coeffs[i] > 0:
                c_nor.append(coeff[i])
                c_abs.append(0)
            else:
                c_nor.append(0)
                c_abs.append(coeff[i])

        plt.figure(figsize=(8, 6))
        plt.bar(inputs.columns, c_nor, width=0.9, color='blue', label='positive')
        plt.bar(inputs.columns, c_abs, width=0.9, color='red', label='negative')
        plt.xticks(rotation=45)
        plt.title('Pareto Representation of ' + target)
        plt.ylabel(target)
        plt.legend()
        plt.show()

# Plot Pareto charts
plot_pareto(coeff_dict, inputs)




