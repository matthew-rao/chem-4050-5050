#Numerical Integration of Second Virial Coefficient

#Import necessary packages
import numpy as np
from scipy.constants import Boltzmann, eV, Avogadro
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import pandas as pd

#Define constants and values
integration_distance = np.linspace(1/1000, 5 * 3.4, 1000)
kB = Boltzmann / eV

#Define potentials for hard-sphere, square-well, and LJ-potentials
def hard_sphere_potential(r, sigma = 3.4):
    """
    Computes the potential energy V(r) for a hard sphere

    Parameters:
    r (float): The distance between 2 particles
    sigma (float): diamater of the hard sphere (assumed to be 3.4 angstroms)

    Returns:
    Float or int: the potential energy (V(r)) between the two Ar atoms.
    """
    if r < sigma:
        return 1000
    else:
        return 0
    
def square_well_potential(r, sigma = 3.4, epsilon = 0.01, lamb = 1.5):
    """
    Computes the potential energy V(r) the square well

    Parameters:
    r (float): The distance between 2 particles
    sigma (float): particle diameter (assumed to be 3.4 angstroms)
    epsilon (float): well depth in eV
    lambda (float): range of well

    Returns:
    Float or int: the potential energy (V(r)) of the square well.
    """
    if r < sigma:
        return 1000
    elif (r >= sigma) and (r < (lamb * sigma)):
        return (-epsilon)
    else:
        return 0
    
def lennard_jones(r, epsilon = 0.01, sigma = 3.4):      #Copied and pasted from optimize_argon_dimer.py
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


#Define B2v function
def compute_b2v(interaction_pot, temperature):
    """
    Computes the second virial potential B2V for each potential at 100K

    Parameters:
    interaction_potential (function): either the hard-sphere, square-well, or LJ potential
    temperature (int or float): temperature in kelvin

    Returns:
    Float: the B2V value for the given potential.
    """
    interaction_potentials = []
    for distance in integration_distance:
        potential = interaction_pot(distance)
        interaction_potentials.append(potential)
    potentials = np.array(interaction_potentials)
    integrand_value = (np.exp((-potentials)/(kB * temperature)) - 1) * (integration_distance ** 2)
    return ((-2 * np.pi * Avogadro) * trapezoid(integrand_value, integration_distance))

#Compute B2v for each potential at 100K
hard_sphere_b2v = compute_b2v(hard_sphere_potential, 100)
square_well_b2v = compute_b2v(square_well_potential, 100)
lennard_jones_b2v = compute_b2v(lennard_jones, 100)

print(hard_sphere_b2v)
print(square_well_b2v)
print(lennard_jones_b2v)

#Initialize variables for varying temp for B2v
temp_range = np.linspace(100, 800, 1000)
hard_sphere_b2v_temp = []
square_well_b2v_temp = []
lj_b2v_temp = []

#B2v for varying temperature in hard sphere potential
for temp in temp_range:
    b2v_temp = compute_b2v(hard_sphere_potential, temp)
    hard_sphere_b2v_temp.append(b2v_temp)

#B2v for varying temperature in square well
for temp in temp_range:
    b2v_temp = compute_b2v(square_well_potential, temp)
    square_well_b2v_temp.append(b2v_temp)

#B2v for varying temperature in LJ
for temp in temp_range:
    b2v_temp = compute_b2v(lennard_jones, temp)
    lj_b2v_temp.append(b2v_temp)


#Plot B2v for all three potentials on the same graph
plt.plot(temp_range, np.array(hard_sphere_b2v_temp), label = "Hard Sphere Potential", color = "red")
plt.plot(temp_range, np.array(square_well_b2v_temp), label = "Square Well", color = "green")
plt.plot(temp_range, np.array(lj_b2v_temp), label = "Lennard Jones Potential", color = "blue")
plt.xlabel("Temperature (Kelvin)")
plt.ylabel("B2v value (Å^3/mol)")
plt.title("Relationship between B2v Value and Varying Temperature")
plt.axhline(y=0, linestyle = "--")
plt.legend()
plt.show()

#Create CSV file
df = pd.DataFrame({
    "Temperature (K)": temp_range,
    "Hard Sphere B2 (Å^3/mol)": hard_sphere_b2v_temp,
    "Square Well B2 (Å^3/mol)": square_well_b2v_temp,
    "Lennard Jones B2 (Å^3/mol)": lj_b2v_temp
})

df.to_csv("B2v_temp.csv", index=False)










