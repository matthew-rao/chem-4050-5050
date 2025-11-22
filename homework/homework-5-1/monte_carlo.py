#Monte Carlo Integration of Hydrogen 2p Orbitals Overlap

#Import necessary packages
import numpy as np
from scipy.stats import expon

#Define the Hydrogen 2p orbital function
def psi_2p_z(x, y, z):
    """
    Computes the psi_2p_z function given cartesian coordinates

    Parameters:
    X (float or int): X coordinate
    Y (float or int): Y coordinate
    Z (float or int): Z coordinate

    Returns:
    psi (Float or int): the value of the wavefunction at cartesian coordinates (x,y,z)
    """

    #Convert from polar coordinates to cartesian coordinates
    r = np.sqrt((x**2) + (y**2) + (z**2))
    cos_theta = z/r

    #Define psi coefficient
    coeff = 1 / (4 * np.sqrt(2 * np.pi))

    #Define wavefunction value
    wavefunction = coeff * (r) * cos_theta * np.exp(-r/2)

    return wavefunction

#Compute overlap integral with random sampling

#Set random seed for reproducibility
np.random.seed(42)

#Generate list of numbers to iterate through
N_list = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]

#Define function for random sampling
def random_sampling(numbers_list, L, R):
    """
    Computes the overlap integral with random sampling

    Parameters:
    numbers (list of ints): list of varying numbers of points to iterate through
    L (int): bound of points to sample from
    R (int or float): separation distance

    Returns:
    integral (list of floats): list of integral calculations of the overlap
    """

    #Create list to append to
    integral_list = []

    #Generate random points with upper and lower bounds, calculate integral
    for numbers in numbers_list:
        x = np.random.uniform(0, L, numbers)
        y = np.random.uniform(0, L, numbers)
        z = np.random.uniform(0, L, numbers)

        integrand = (psi_2p_z(x, y, z + R/2)) * (psi_2p_z(x, y, z - R/2))
        integral = ((2 * L)**3) * np.mean(integrand)    #Multiply by (2L)^3 to account for 8 octants
        integral_list.append(integral)

    return integral_list

#Compute overlap integral with importance sampling
def importance_sampling(numbers_list, R):
    """
    Computes the overlap integral with importance sampling

    Parameters:
    numbers (list of ints): list of varying numbers of points to iterate through
    R (int or float): separation distance

    Returns:
    integral (list of floats): list of integral calculations of the overlap
    """

    #Create list to append to
    integral_list = []

    #Generate random points with upper and lower bounds, calculate integral
    for numbers in numbers_list:
        x = expon.rvs(size=numbers, scale=1)
        y = expon.rvs(size=numbers, scale=1)
        z = expon.rvs(size=numbers, scale=1)

        numer = psi_2p_z(x,y,z + (R/2)) * psi_2p_z(x,y,z - (R/2))
        denom = expon.pdf(x) * expon.pdf(y) * expon.pdf(z)

        integrand = numer / denom
        integral = 8 * np.mean(integrand)   #Multiple by 8 to account for 8 octants
        integral_list.append(integral)

    return integral_list

#Plot overlap integral as a function of separation distance using importance sampling
importance_points = [1000000]
r_list = np.arange(0.5, 20, 0.5)

separation_overlaps =[]
for r in r_list:
    integral = importance_sampling(importance_points, r)[0]
    separation_overlaps.append(integral)


#Run our functions
randomsampling_sr = random_sampling(N_list, 20, 2)
importancesampling_sr = importance_sampling(N_list, 2)