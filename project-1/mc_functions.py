#Grand Canonical Monte Carlo Simulation

#Import necessary packages
import numpy as np
import matplotlib.pyplot as plt

#Initialize lattice function
def initialize_lattice(size):
    """
    Creates a 2D square lattice of dimension "size" with initialized value 0

    Parameters:
    size (int): dimension of square lattice

    Returns:
    lattice (numpy array): empty 2D square lattice of dimension size
    """
    lattice = np.zeros([size, size])
    return lattice

#Compute the indices of the neighbor with periodic boundaray conditions
def compute_neighbor_indices(size):
    """
    Computes the coordinates of the four neighbors for the given space in the square lattice

    Parameters:
    size (int): dimension of square lattice

    Returns:
    neighbor_indices (dict): dictionary with the index coordinates as keys and a list containing neighboring indices as values
    """
    neighbor_indices = {}

    #Nested for loop looping over size
    for x in range(size):
        for y in range(size):
            neighbors = [
                ((x - 1) % size, y),    #Find neighboring indices by using the modulo function at the edges
                ((x + 1) % size, y),
                (x, (y - 1) % size),
                (x, (y + 1) % size)
            ]
            neighbor_indices[(x, y)] = neighbors
    return neighbor_indices

#Calculate Interaction Energy
def calculate_interaction_energy(lattice, site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB):
    """
    Calculate the interaction energy of a given particle in the lattice

    Parameters:
    lattice(numpy array): 2D array of dimension "size" indicating which sites are occupied
    site (numpy array): numpy coordinates of the site being examined
    particle (int): particle being examined
    neighbor_indices (dict): dictionary with the index coordinates as keys and a list containing neighboring indices as values
    epsilon_AA (int or float): interaction energy between 2 A particles
    epsilon_BB (int or float): interaction energy between 2 B particles
    epsilon_AB (int or float): interaction energy between A and B particles

    Returns:
    interaction_energy (int or float): interaction energy of a given particle
    """
    #Initialize site and interaction energy
    x,y = site
    interaction_energy = 0

    #Loop over neighbor indices to calculate potential energy
    for neighbor in neighbor_indices[(x,y)]:
        neighbor_particle = lattice[neighbor]
        if neighbor_particle != 0:
            if particle == 1:   #Particle is Particle A
                if neighbor_particle == 1:
                    interaction_energy += epsilon_AA
                else:   #Neighbor is particle B
                    interaction_energy += epsilon_AB
            else:   #Particle is Particle B
                if neighbor_particle == 2:  #Neighbor is Particle B
                    interaction_energy += epsilon_BB
                else:   #Neighbor is Particle A
                    interaction_energy += epsilon_AB
    return interaction_energy

#Attempt to Add or Remove a Particle
def attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params):
    """
    Attempts to add or remove a particle based on randomness

    Parameters:
    lattice(numpy array): 2D array of dimension "size" indicating which sites are occupied
    N_A (int): number of A particles present in lattice
    N_B (int): number of B particles present in lattice
    N_empty (int): number of empty positions in lattice
    neighbor_indices (dict): dictionary with the index coordinates as keys and a list containing neighboring indices as values
    params (dict): dictionary containing labels as keys and temperature, chemical potentials, and interaction energies as values

    Returns:
    N_A (int): number of A particles present in lattice
    N_B (int): number of B particles present in lattice
    N_empty (int): number of empty positions in lattice
    """
    #Initialize size, number of sites, and beta
    size = np.shape(lattice)[0]
    N_sites = size * size
    beta = 1 / params["T"]

    #Extract our parameters from param
    epsilon_A = params["epsilon_A"]
    epsilon_B = params["epsilon_B"]
    epsilon_AA = params["epsilon_AA"]
    epsilon_BB = params["epsilon_BB"]
    epsilon_AB = params["epsilon_AB"]
    mu_A = params["mu_A"]
    mu_B = params["mu_B"]

    #Randomly decide whether or not to add or remove particle (50% chance each)
    if np.round(np.random.rand()):  #Round to 1 if yes, round to 0 if no
        if N_empty == 0:
            return N_A, N_B, N_empty    #No empty sites available
        
        empty_sites = np.argwhere(lattice == 0) #Find empty sites in lattice
        random_index = np.random.randint(np.shape(empty_sites)[0])
        x, y = empty_sites[random_index] #Get coordinates
        site = (x,y)

        #Decide to randomly add A or B
        if np.round(np.random.rand()):
            particle = 1
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A
        else:   #Add B
            particle = 2
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B
        #Calculate Delta E
        delta_E = epsilon + calculate_interaction_energy(lattice, site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB)
        #Calculate acceptance probability
        acc_prob = min(1, (N_empty) / (N_s + 1) * np.exp(-beta * (delta_E - mu)))
        r = np.random.rand()
        if r < acc_prob:
            lattice[site] = particle
            if particle == 1:
                N_A += 1
            else:
                N_B += 1
            N_empty -= 1
    else:   #Remove a particle
        if (N_sites - N_empty == 0):
            return N_A, N_B, N_empty    #No particles to remove
        
        occupied_sites = np.argwhere(lattice != 0)  #Find occupied sites in lattice
        random_index = np.random.randint(np.shape(occupied_sites)[0])
        x, y = occupied_sites[random_index] #Get coordinates
        site = (x,y)
        particle = lattice[site]

        #Initialize particle values
        if particle == 1:   #Particle A
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A
        else:   #Particle B
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B
        
        #Calculate delta E
        delta_E = -epsilon - calculate_interaction_energy(lattice, site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB)

        #Calculate acceptance probability
        acc_prob = min(1, N_s / (N_empty + 1) * np.exp(-beta * (delta_E + mu)))

        r = np.random.rand()
        if r < acc_prob:
            lattice[site] = 0 #Remove particle
            if particle == 1:
                N_A -= 1
            else:
                N_B -= 1
            N_empty += 1
    return N_A, N_B, N_empty

#Run the GCMC Simulation
def run_simulation(size, n_steps, params):
    """
    Run the Monte Carlo simulation

    Parameters:
    size (int): dimension of our square lattice
    n_steps (int): number of steps to traverse in the given simulation
    params (dict): dictionary containing labels as keys and temperature, chemical potentials, and interaction energies as values

    Returns:
    lattice (numpy array): 2D array of dimension "size" indicating which sites are occupied
    coverage_A (numpy array): proportion of sites occupied by A for each step
    coverage_B (numpy array): proportion of sites occupied by B for each step
    """

    #Initialize lattice
    lattice = initialize_lattice(size)

    #Compute neighbor indices
    neighbor_indices = compute_neighbor_indices(size)
    N_sites = size * size

    #Initialize counts
    N_A = 0
    N_B = 0
    N_empty = N_sites

    #Create arrays
    coverage_A = np.zeros(n_steps)
    coverage_B = np.zeros(n_steps)

    # Record lattice at every step
    lattice_history = np.zeros((n_steps, size, size), dtype=int)

    #Update arrays
    for step in range(n_steps):
        N_A, N_B, N_empty = attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params)
        coverage_A[step] = N_A / N_sites
        coverage_B[step] = N_B / N_sites
    return lattice, coverage_A, coverage_B

#Plot Lattice Configuration
def plot_lattice(lattice, ax, title):
    """
    Plot the lattice and its coverage

    Parameters:
    lattice (numpy array): 2D array of dimension "size" indicating which sites are occupied
    ax (plt object): axes to display lattice
    title (string): title of plot

    Returns:
    ax (plt object): axis of plotted lattice
    """
    #Initialize size
    size = np.shape(lattice)[0]

    #Loop through size to plot
    for x in range(size):
        for y in range(size):
            if lattice[x,y] == 1:
                ax.plot(x + 0.5, y + 0.5, "ro") #Red represents hydrogen
            elif lattice[x,y] == 2:
                ax.plot(x + 0.5, y + 0.5, "bo") #Blue represents nitrogen
    #Set axis limits and labels
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_yticks(np.arange(0, size, 1))
    ax.set_xticks(np.arange(0, size, 1))
    ax.grid()
    ax.tick_params(axis = 'both', which = 'both', labelbottom = False, labelleft = False)

    #Set title
    ax.set_title(title, fontsize=10)
    return ax

