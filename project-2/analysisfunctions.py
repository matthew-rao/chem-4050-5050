#Define functions for analysis

#Define constants
mass = 1.0 #set to arbitrary units for simplicity
sigma = 1.0 #characteristic length scale in the Lennard-Jones potential, set to 1.
kB = 1.0 #Boltzmann constant, set to 1 for reduced units
k = 1.0  # Spring constant
n_particles = 20  # Number of particles
epsilon_repulsive = 1.0  # Depth of repulsive LJ potential
epsilon_attractive = 0.5  # Depth of attractive LJ potential
cutoff = 2**(1/6) * sigma # Cutoff distance
dt = 0.01  # Time step
box_size = 100.0  # Size of the cubic box
r0 = 1.0  # Equilibrium bond length

#Import necessary packages
import numpy as np
import mdsim

#Write function to calculate radius of gyration
def calculate_radius_of_gyration(positions):
    """
    Calculate radius of gyration for the particles on the chain
    
    Parameters:
    positions (numpy array): array of all positions of the particles on the chain

    Returns:
    Rg (int or float): radius of gyration for given input array
    """ 
    center_of_mass = np.mean(positions, axis = 0)
    Rg_squared = np.mean(np.sum((positions - center_of_mass)**2, axis=1))
    Rg = np.sqrt(Rg_squared)
    return Rg

#Write function to calculate end-to-end distances
def calculate_end_to_end_distance(positions):
    """
    Calculate the end-to-end distance of the entire polymer
    
    Parameters:
    positions (numpy array): array of all positions of the particles on the chain

    Returns:
    Ree (int or float): total distance from end-to-end
    """
    Ree = np.linalg.norm(positions[-1] - positions[0])
    return Ree 

#Write function to calculate harmonic potential energies
def calculate_harmonic_pe(positions, k):
    """
    Calculate the harmonic bond potential energy of the entire polymer
    
    Parameters:
    positions (numpy array): array of all positions of the particles on the chain
    k (int or float): spring constant

    Returns:
    harmonic_pe (int or float): harmonic bond potential energy
    """
    #Initialize value for harmonic_pe
    harmonic_pe = 0
    for i in range(n_particles - 1):
        displacement = positions[i + 1] - positions[i]
        displacement = mdsim.minimum_image(displacement, box_size)
        distance = np.linalg.norm(displacement)
        harmonic_pe += 0.5 * k * (distance - r0)**2 #Increment harmonic_pe with each potential energy calculation
    return harmonic_pe

#Write function to calculate LJ potential energies
def calculate_lj_pe(positions, interaction_type):
    """
    Calculate the overall Lennard Jones potential energy of the entire polymer
    
    Parameters:
    positions (numpy array): array of all positions of the particles on the chain
    epsilon (int or float): well depth of LJ potential
    interaction_type (string): type of interaction present between particles

    Returns:
    lj_pe (int or float): Lennard Jones potential energy
    """

    #Initialize our potential energy
    lj_pe = 0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            #Calculate distance
            displacement = positions[j] - positions[i]
            displacement = mdsim.minimum_image(displacement, box_size)
            distance = np.linalg.norm(displacement)
            #Check interaction type and update potential energy accordingly
            if interaction_type == "repulsive" and np.abs(i - j) == 2:
                if distance < cutoff:
                    lj_pe += 4 * epsilon_repulsive * ((sigma/distance)**12 - (sigma/distance)**6 + 0.25)
            elif interaction_type == "attractive" and np.abs(i - j) > 2:
                lj_pe += 4 * epsilon_attractive * ((sigma/distance)**12 - (sigma/distance)**6)
            else:
                continue
    return lj_pe