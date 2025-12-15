# MD Simulation of Polymer Chain

#Import necessary packages
import numpy as np

#Initialize constant values
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

#Create function to initialize chain
def initialize_chain(n_particles, box_size, r0):
    """
    Initializes the polymer, or "snake", inside of a box. 

    Parameters:
    n_particles (int): number of particles in chain
    box_size (int or float): dimension of simulation box
    r0 (int or float): equally spaced distance between each particle

    Returns:
    positions (numpy array): array of all positions of the particles on the chain
    """
    #Create positions array and initialize the first coordinate to be the box's center
    positions = np.zeros([n_particles, 3])
    current_position = [box_size / 2, box_size / 2, box_size / 2]
    positions[0] = current_position

    #Generate random unit vector and adjust position of new coordinate based on unit vector
    for i in range(n_particles):
        vector = np.random.rand(3)
        direction = vector / np.linalg.norm(vector)
        next_position = current_position + r0 * direction
        positions[i] = apply_pbc(next_position, box_size)
        current_position = positions[i]
    return positions

#Create function to initialize
def initialize_velocities(n_particles, target_temperature, mass):
    """
    Initializes the velocities of the particles inside the box

    Parameters:
    n_particles (int): number of particles in chain
    target_temperature (int or float): temperature of simulation run
    mass (int or float): mass of each particle

    Returns:
    velocities (numpy array): array of all velocities of the particles on the chain
    """
    #Sample random velocities from the Maxwell-Boltzmann distribution, generate random numbers from dist
    #Use mean 0 and std of square root(kB * T/m)
    velocities = np.random.normal(0, np.sqrt((kB * target_temperature) / mass), (n_particles, 3))
    velocities -= np.mean(velocities, axis=0) # Remove net momentum
    return velocities

#Create function to apply periodic boundary conditions
def apply_pbc(position, box_size):
    """
    Applies periodic boundary conditions to the box by taking the modulus of position with respect to box size

    Parameters:
    position (np array): coordinates of the position of the particle
    box_size (int or float): dimension of simulation box

    Returns:
    numpy array: coordinates of updated position
    """
    return position % box_size

#Write function to compute the minimum image
def minimum_image(displacement, box_size):
    """
    Compute the shortest vector between two particles under periodic boundary conditions

    Parameters:
    displacement (np array): array of displacement between particles and the vector between them
    box_size (int or float): dimension of simulation box

    Returns:
    updated_displacement (numpy array): updated array to account for minimum image
    """   
    updated_displacement = displacement - box_size * np.round(displacement / box_size)
    return updated_displacement

#Write function to compute harmonic forces
def compute_harmonic_forces(positions, k, r0, box_size):
    """
    Compute harmonic force between adjacent particles

    Parameters:
    positions (numpy array): array of all positions of the particles on the chain
    k (int or float): spring constant
    r0 (int or float): equally spaced distance between each particle
    box_size (int or float): dimension of simulation box

    Returns:
    forces (numpy array): forces felt by all particles on the polymer
    """
    #Initialize array with the same shape as positions
    forces = np.zeros_like(positions)
    for i in range(n_particles - 1):
        displacement = positions[i + 1] - positions[i]
        displacement = minimum_image(displacement, box_size)
        distance = np.linalg.norm(displacement)
        #Calculate magnitude of forces
        force_magnitude = -k * (distance - r0)
        force = force_magnitude * (displacement / distance)
        #Update forces within the array
        forces[i] -= force
        forces[i + 1] += force
    return forces

#Write function to compute Lennard-Jones forces
def compute_lennard_jones_forces(positions, epsilon, sigma, box_size, interaction_type):
    """
    Compute Lennard Jones forces, either attractive or repulsive, from nonadjacent particles

    Parameters:
    positions (numpy array): array of all positions of the particles on the chain
    epsilon (int or float): well depth of LJ potential
    sigma (int or float): separation potential in LJ
    box_size (int or float): dimension of simulation box
    interaction_type (string): type of interaction present between particles

    Returns:
    forces (numpy array): forces felt by all particles on the polymer
    """

    #Initialize array with the same shape as positions
    forces = np.zeros_like(positions)
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            #Check distance of particles and type of interaction
            if interaction_type == "repulsive" and np.abs(i - j) == 2:
                epsilon = epsilon_repulsive
            elif interaction_type == "attractive" and np.abs(i - j) > 2:
                epsilon = epsilon_attractive
            else:
                continue
            #Calculate displacement
            displacement = positions[j] - positions[i]
            displacement = minimum_image(displacement, box_size)
            distance = np.linalg.norm(displacement)
            if distance < cutoff:
                #Compute LJ force and update the array
                force_magnitude = 24 * epsilon * ((sigma / distance)**(12) - 0.5 * (sigma / distance)**(6)) / distance
                force = force_magnitude * (displacement / distance)
                forces[i] -= force
                forces[j] += force
    return forces

#Write function to compute velocity verlet integration
def velocity_verlet(positions, velocities, forces, dt, mass):
    """
    Compute the positions and velocities of the updated polymer particles using velocity verlet integration

    Parameters:
    positions (numpy array): array of all positions of the particles on the chain
    velocities (numpy array): array of all velocities of the particles on the chain
    forces (numpy array): forces felt by all particles on the polymer
    dt (float): timestep of simulation
    mass (int or float): mass of each particle

    Returns:
    positions (numpy array): array of all updated positions of the particles on the chain
    velocities (numpy array): array of all updated velocities of the particles on the chain
    forces_new (numpy array): updated forces felt by all particles on the polymer
    """
    #Compute velocities and update positions accordingly
    velocities += 0.5 * forces / mass * dt
    positions += velocities * dt
    positions = apply_pbc(positions, box_size)
    #Compute forces and update velocities accordingly
    new_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
    new_lj_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, "repulsive")
    new_lj_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, "attractive")
    forces_new = new_harmonic + new_lj_repulsive + new_lj_attractive
    velocities += 0.5 * forces_new / mass * dt
    return positions, velocities, forces_new

#Write function to rescale velocities
def rescale_velocities(velocities, target_temperature, mass):
    """
    Rescale velocities to fit Maxwell Boltzmann Distribution
    
    Parameters:
    velocities (numpy array): array of all velocities of the particles on the chain
    target_temperature (int or float): temperature of simulation run
    mass (int or float): mass of each particle

    Returns:
    velocities (numpy array): array of all updated velocities of the particles on the chain
    """ 
    #Compute kinetic energy
    kinetic_energy = 0.5 * mass * sum(np.linalg.norm(velocities, axis = 1)**2)
    #Adjust temperature
    current_temperature = (2/3) * kinetic_energy / (n_particles * kB)
    scaling_factor = np.sqrt(target_temperature / current_temperature)
    #Scale velocities by scaling factor
    velocities *= scaling_factor
    return velocities

 





        

