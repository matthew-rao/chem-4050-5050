# Part 2: Argon Trimer

#Import necessary packages and past functions
import numpy as np
from optimize_argon_dimer import lennard_jones
from scipy.optimize import minimize
from math import sqrt

#Copy and Paste HW1 Functions
def compute_bond_length(coord1, coord2):
    """Takes the coordinates of 2 atoms and returns the bond length between the atoms,
        where coord1 and coord2 represent 2 lists of Cartesian coordinates.

        If the bond length returned is greater than 2 angstroms, a string warning will be printed
        alongside the float or integer bond length
    """
    
    array1 = np.array(coord1)                               #Convert to numpy
    array2 = np.array(coord2)

    coordinate_difference = array1 - array2
    squared_difference = coordinate_difference ** 2
    bond_length = np.sqrt(np.sum(squared_difference))       #Use distance formula to calculate Euclidian distance

    #Commenting this section out to reduce clutter in output
    # if bond_length < 2:
    #     print(f"Bond length: {bond_length} angstroms")
    # else:
    #     print(f"The bond length is: {bond_length} angstroms, which does not fall in the covalent bond length range.")

    return bond_length

def compute_bond_angle(coord1, coord2, coord3):
    """Takes the Cartesian coordinates of 3 atoms and returns the bond angle in degrees,
        where coord1, coord2, and coord3 represent lists of float or integer values.
        
        Function will also print the classification of the angle as right, obtuse or acute.
        """
    
    a = np.array(coord1)                                    #Convert to numpy
    b = np.array(coord2)
    c = np.array(coord3)

    ab = a - b
    bc = c - b                                              #Find vectors AB and BC
    numerator = np.dot(ab, bc)                              #Dot product between given vectors             

    ab_magnitude = np.linalg.norm(ab)                       
    bc_magnitude = np.linalg.norm(bc)
    denominator = ab_magnitude * bc_magnitude               #Find product of the magnitude of the vectors

    bond_angle = np.degrees(np.arccos(numerator / denominator))

    #Commenting this section out to reduce clutter in output
    # if bond_angle > 90:
    #     print(f"Bond angle: {bond_angle}. It is obtuse")

    # elif bond_angle < 90:
    #     print(f"Bond angle: {bond_angle}. It is acute")
    # else:
    #     print(f"Bond angle: {bond_angle}. It is a right angle")

    return bond_angle

#Define potential energy function for trimer using LJ equation
def trimer_potential(coordinates):
    """
    Computes the potential energy V(r) between 3 Ar atoms (a trimer).

    Parameters:
    coordinates (list): containing 3 variables:
    r12 (float): distance between atoms 1 and 2 (the x-coordinate of atom 2)
    x3 (float): x-coordinate of atom 3
    y3 (float): y-coordinate of atom 3

    Returns:
    Float: the potential energy (V(r)) between the three Ar atoms.
    """

    #Establish the coordinates of our three atoms
    atom1 = [0,0]
    atom2 = [coordinates[0], 0]
    atom3 = [coordinates[1], coordinates[2]]

    #Find bond lengths
    r12 = coordinates[0]
    r13 = compute_bond_length(atom1, atom3)
    r23 = compute_bond_length(atom2, atom3)

    #Find individual potential energies
    lj12 = lennard_jones(r12)
    lj13 = lennard_jones(r13)
    lj23 = lennard_jones(r23)

    total_lj = lj12 + lj13 + lj23
    return total_lj

#Minimize trimer_potential function. Assume equilaterial triangle for starting guess
minimized_trimer = minimize(trimer_potential, [3, 1.5, (3 * sqrt(3) / 2)], method='nelder-mead').x
print(minimized_trimer) #r12 (x2) = 3.81640722, x3 = 1.9081822, y3 = 3.30510345

#Calculate and print the optimized distances
r12 = 3.81640722
r13 = compute_bond_length([0,0], [1.9081822, 3.30510345])
r23 = compute_bond_length([3.81640722, 0], [1.9081822, 3.30510345])

print(f"The optimized distances are r12 = {r12:.4f}, r13 = {r13:.4f}, r23 = {r23:.4f}")

#Calculate and print the optimal angles
atom1 = [0,0]
atom2 = [3.81640722, 0]
atom3 = [1.9081822, 3.30510345]
angle123 = compute_bond_angle(atom1,atom2,atom3)
angle231 = compute_bond_angle(atom2,atom3,atom1)
angle312 = compute_bond_angle(atom3,atom1,atom2)

print(f"The optimized angles are angle 1 = {angle312:.4f}, angle 2 = {angle123:.4f}, angle 3 = {angle231:.4f}")
#Each angle is approximately 60 degrees and the bond length is the same

#Comment on geometric arrangement
print("The atoms each have the same bond length from eaach other. " \
"Each angle is also 60 degrees. This implies an equilateral triangle arrangement")

#Make XYZ file
atoms = [
        ("Ar", 0.0000, 0.0000),
        ("Ar", 3.8164, 0.0000),
        ("Ar", 1.9082, 3.3051),
    ]

with open("argon.xyz", 'w') as xyz_file:
    xyz_file.write(f"3\n")
    xyz_file.write(f"Argon trimer\n")
    for atom, x, y in atoms:
        xyz_file.write(f"{atom} {x:.4f} {y:.4f}\n")


