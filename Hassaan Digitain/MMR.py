# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:42:03 2024

@author: Noman Ajmal

This program calculate G1, alpha, & G2 
using the following formulas

G1 = (P^2*a^3)/(2*B*E*I)
alpha = (G1*B*E)/(6*P^2*a)
G2 = (6*P^2*a)/(B*E*(1+alpha))

where:
    P = force
    a = crack length
    B = width of sample
    E = material property
    I = geometric FACTOR
    
    alpha = mode mixture transformation parameter for setting lever length
    G1 = opening (Mode I) component of strain energy release
        rate, kJ/m2
    G2 = shear (Mode II) component of strain energy release
         rate, kJ/m2

"""

def calculate_G1(P, a, B, E, I):
    try:
        G1 = (P**2 * a**3) / (2 * B * E * I)
    except ZeroDivisionError:
        print("Error: Division by zero in G1 calculation.")
        return None
    except OverflowError:
        print("Overflow error in G1 calculation.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in G1 calculation: {e}")
        return None
    return G1

def calculate_alpha(G1, P, a, B, E):
    try:
        alpha = (G1 * B * E) / (6 * P**2 * a)
    except ZeroDivisionError:
        print("Error: Division by zero in alpha calculation.")
        return None
    except OverflowError:
        print("Overflow error in alpha calculation.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in alpha calculation: {e}")
        return None
    return alpha
    
def calculate_G2(P, a, B, E, alpha):
    try:
        G2 = (6 * P**2 * a) / (B * E * (1 + alpha))
    except ZeroDivisionError:
        print("Error: Division by zero in G2 calculation.")
        return None
    except OverflowError:
        print("Overflow error in G2 calculation.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in G2 calculation: {e}")
        return None
    return G2


def main(P, a, B, E, I):
    G1 = calculate_G1(P, a, B, E, I)
    if G1 is None:
        print("Failed to calculate G1.")
        return

    alpha = calculate_alpha(G1, P, a, B, E)
    if alpha is None:
        print("Failed to calculate alpha.")
        return

    G2 = calculate_G2(P, a, B, E, alpha)
    if G2 is None:
        print("Failed to calculate G2.")
        return

    print(f"Calculated values:\nG1: {G1}\nAlpha: {alpha}\nG2: {G2}")

# Example Parameters
P = 500  # Example force value in Newtons (N)
a = 0.05  # Example crack length in meters (m)
B = 0.02  # Example width of sample in meters (m)
E = 70e9  # Example material property (Young's Modulus) in Pascals (Pa)
I = 1e-6  # Example geometric factor

# Run the main function
main(P, a, B, E, I)
