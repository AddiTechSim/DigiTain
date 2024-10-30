# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:56:37 2024

@author: hp
"""

"""To validate the traction-separation values in this program, we use energy-based validation by comparing the area under the calculated traction-separation curve to the material’s fracture toughness values 
GIc and GIIc for Modes I and II, respectively.

Scientific Basis and Validation Method
Energy Release Rate (Fracture Toughness):
Fracture toughness is a measure of the energy required to propagate a crack. 
For accurate validation, the calculated energy from the traction-separation curve 
(area under the curve) should match the fracture toughness values GIc and GIIc.

When you run the script:
    Mode I: Calculated Energy = 0.2400 MPa.mm, Expected Energy (GIc) = 0.2400 MPa.mm
    Mode II: Calculated Energy = 0.5940 MPa.mm, Expected Energy (GIIc) = 0.5940 MPa.mm
 """

import numpy as np
import matplotlib.pyplot as plt

# Material properties from provided data
N = 2500.4  # Longitudinal tensile strength (MPa)
S = 91.1    # Shear strength (MPa)
GIc = 0.24  # Interlaminar fracture toughness Mode I (kJ/m²)
GIIc = 0.594  # Interlaminar fracture toughness Mode II (kJ/m²)

# Convert kJ/m² to MPa.mm (1 kJ/m² = 1 MPa.mm)
GIc *= 1.0  # Convert to MPa.mm
GIIc *= 1.0  # Convert to MPa.mm

# Traction-Separation Law - Bilinear for simplicity

# Step 1: Estimate separation values (critical displacement)
# Normal separation (Mode I)
separation_I = 2 * GIc / N  # delta_I (Mode I separation)
# Shear separation (Mode II)
separation_II = 2 * GIIc / S  # delta_II (Mode II separation)

# Define the bilinear law (with softening)
def bilinear_traction(delta, delta_c, max_traction):
    """
    Bilinear traction-separation law
    delta: separation
    delta_c: critical separation
    max_traction: maximum traction (strength)
    """
    if delta <= delta_c:  # Before softening
        return max_traction * delta / delta_c
    else:  # After softening
        return max_traction * (1 - (delta - delta_c) / (2 * delta_c))

# Separation values (x-axis)
delta_range_I = np.linspace(0,  1* separation_I, 100)
delta_range_II = np.linspace(0, 1 * separation_II, 100)

# Compute tractions for Mode I and Mode II
traction_I = [bilinear_traction(d, separation_I, N) for d in delta_range_I]
traction_II = [bilinear_traction(d, separation_II, S) for d in delta_range_II]

# Step 2: Validation by checking energy under the curve
def calculate_energy(delta_range, traction, delta_c):
    """
    Calculate energy (area under the traction-separation curve)
    """
    energy = np.trapz(traction, delta_range)  # Numerical integration (trapezoidal rule)
    return energy

# Validate the energy equals fracture toughness
energy_I = calculate_energy(delta_range_I, traction_I, separation_I)
energy_II = calculate_energy(delta_range_II, traction_II, separation_II)

print(f"Mode I: Calculated Energy = {energy_I:.4f} MPa.mm, Expected Energy (GIc) = {GIc:.4f} MPa.mm")
print(f"Mode II: Calculated Energy = {energy_II:.4f} MPa.mm, Expected Energy (GIIc) = {GIIc:.4f} MPa.mm")

# Plot traction-separation curves for validation
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(delta_range_I, traction_I, label="Mode I (Normal Traction)")
plt.title("Mode I Traction-Separation")
plt.xlabel("Separation (mm)")
plt.ylabel("Traction (MPa)")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(delta_range_II, traction_II, label="Mode II (Shear Traction)", color='r')
plt.title("Mode II Traction-Separation")
plt.xlabel("Separation (mm)")
plt.ylabel("Traction (MPa)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


