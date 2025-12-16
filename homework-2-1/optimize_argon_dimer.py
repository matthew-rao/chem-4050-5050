# Part 1: Argon Dimer

#Defines the LJ Function
def lennard_jones(r, epsilon = 0.01, sigma = 3.4):
    """
    Computes the potential energy V(r) between 2 Ar atoms.

    Parameters:
    r (float): The distance between 2 Ar atoms.
    epsilon (float): The potential well depth (minimum energy)
    sigma (float): Distance at which potential is zero.

    Returns:
    Float: the potential energy (V(r)) between the two Ar atoms.
    """

    potential = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

    return potential

#Import necessary packages
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

#Find value that minimizes LJ Potential
minimized_LJ = minimize(lennard_jones, 4, method='nelder-mead').x[0]   #Starting guess of 4 angstroms
print(f"The distance that minimizes the LJ potential is {minimized_LJ:.4f} angstroms")

#Plot the Potential Energy Curve
x_values = np.linspace(3, 6, 100)
y_values = []
for x in x_values:
    potential = lennard_jones(x)
    y_values.append(potential)

plt.plot(x_values, y_values)
plt.plot(minimized_LJ, lennard_jones(minimized_LJ), 'ro', label = "Equilibrium Distance")
plt.legend()
plt.xlabel("Interatomic Distance Between Ar (angstroms)")
plt.ylabel("Potential Energy V(r)")
plt.title("Lennard-Jones Potential for Argon Dimer")
plt.show()

#Make XYZ file
atoms = [
        ("Ar", 0.0000, 0.0000),
        ("Ar", minimized_LJ, 0.0000)
        ]

with open("argondimer.xyz", 'w') as xyz_file:
    xyz_file.write(f"2\n")
    xyz_file.write(f"Argon dimer\n")
    for atom, x, y in atoms:
        xyz_file.write(f"{atom} {x:.4f} {y:.4f}\n")

