#Thermodynamic Processes of Ce(3)

#Import necessary packages
from scipy.constants import k, eV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Define constants
kB = k / eV

#Define partition function for Isolated Ce(3)
def partition_isolated(g, T):
    """
    Computes the partition function for an isolated Ce3+ ion with 14-fold degeneracy - all states have zero energy

    Parameters:
    g (int): the degeneracy of the Ce3+ ion
    T (numpy array): an array of the given temperatures that we want to examine

    Returns:
    Numpy array: an array of the partition function, Z, as a function of varying temperature
    """
    partition = g * np.exp((-0) / (kB * T))
    return partition

#Define partition function for Ce(3) with spin-orbin coupling (SOC)
def partition_SOC(T):
    """
    Computes the partition function for Ce3+ with SOC

    Parameters:
    T (numpy array): an array of the given temperatures that we want to examine

    Returns:
    Numpy array: an array of the partition function, Z, as a function of varying temperature
    """

    #First partition calculation, g = 6 (E=0)
    first_part = 6 * np.exp((-0) / (kB * T))

    #Second partition calculation, 0.28 eV energy difference (g = 8)
    second_part = 8 * np.exp((-0.28)/ (kB * T))

    return first_part + second_part

#Define partition function of Ce(3) with SOC and CFS
def partition_SOC_CFS(T):
    """
    Computes the partition function for Ce3+ with SOC

    Parameters:
    T (numpy array): an array of the given temperatures that we want to examine

    Returns:
    Numpy array: an array of the partition function, Z, as a function of varying temperature
    """

    #First partition calculation, ground state, g = 4
    first_p = 4 * np.exp((-0) / (kB * T))

    #Second partition, g = 2, difference = 0.12
    second_p = 2 * np.exp((-0.12) / (kB * T))

    #Third partition, g = 2, difference (from ground) = 0.25
    third_p = 2 * np.exp((-0.25) / (kB * T))

    #Fourth partition, g = 4, difference (from ground) = 0.32
    fourth_p = 4 * np.exp((-0.32) / (kB *T))

    #Fifth partition, g = 2, difference (from ground) = 0.46
    fifth_p = 2 * np.exp((-0.46) / (kB * T))

    return first_p + second_p + third_p + fourth_p + fifth_p

#Define functions to calculate thermodynamic properties
#Define function to calculate internal energy
def internal_e(Z, T):
    """
    Computes the internal energy

    Parameters:
    Z (numpy array): an array of the partition functions previously calculated
    T (numpy array): an array of the given temperatures that we want to examine

    Returns:
    Numpy array: an array of internal energy, U, as a function of varying temperature
    """
    E_avg = -np.gradient(np.log(Z), 1 / (kB * T))   #Where the average energy is a measure of internal energy
    return E_avg

#Define function to calculate free energy
def free_e(Z, T):
    """
    Computes the free energy

    Parameters:
    Z (numpy array): an array of the partition functions previously calculated
    T (numpy array): an array of the given temperatures that we want to examine

    Returns:
    Numpy array: an array of free energy, F, as a function of varying temperature
    """
    free_e = -kB * T * np.log(Z)   #Where the average energy is a measure of internal energy
    return free_e

#Define function to calculate entropy
def entropy(F, T):
    """
    Computes the entropy of system

    Parameters:
    F (numpy array): an array of the free energy previously calculated
    T (numpy array): an array of the given temperatures that we want to examine

    Returns:
    Numpy array: an array of entropy, S, as a function of varying temperature
    """
    S = -np.gradient(F, T)   #Where the average energy is a measure of internal energy
    return S

#Initialize temperatures that we will be examining
temps = np.linspace(300, 2000, 2000)

#Isolated Ce(3) - Compute thermodynamic processes for this
isolated_p = partition_isolated(14, temps)
isolated_U = internal_e(isolated_p, temps)
isolated_F = free_e(isolated_p, temps)
isolated_entropy = entropy(isolated_F, temps)

#Ce(3) with SOC - Compute the thermodynamic processes for this
soc_p = partition_SOC(temps)
soc_U = internal_e(soc_p, temps)
soc_F = free_e(soc_p, temps)
soc_entropy = entropy(soc_F, temps)

#Ce(3) with SOC and CFS - Compute the thermodynamic processes for this
soc_cfs_p = partition_SOC_CFS(temps)
soc_cfs_U = internal_e(soc_cfs_p, temps)
soc_cfs_F = free_e(soc_cfs_p, temps)
soc_cfs_entropy = entropy(soc_cfs_F, temps)

#Create CSV file
df = pd.DataFrame({
    "Temperature (K)": temps,
    "U (isolated)": isolated_U,
    "F (isolated)": isolated_F,
    "S (isolated)": isolated_entropy,
    "U (soc)": soc_U,
    "F (soc)": soc_F,
    "S (soc)": soc_entropy,
    "U (soc_cfs)": soc_cfs_U,
    "F (soc_cfs)": soc_cfs_F,
    "S (soc_cfs)": soc_cfs_entropy,
})

df.to_csv("ce_thermo.csv", index=False)

#Plot figures

#Plot Internal Energy
plt.plot(temps, isolated_U, label = "Isolated Ce(3)", color = "red")
plt.plot(temps, soc_U, label = "Ce(3) with SOC", color = "blue")
plt.plot(temps, soc_cfs_U, label = "Ce(3) with SOC and CFS", color = "green")
plt.xlabel("Temperature (K)")
plt.ylabel("Internal Energy (eV)")
plt.title("Relationship Between Internal Energy and Temperature")
plt.legend()
plt.savefig("internalenergy.png", dpi=300, bbox_inches="tight")
plt.show()

#Plot Free Energy
plt.plot(temps, isolated_F, label = "Isolated Ce(3)", color = "red")
plt.plot(temps, soc_F, label = "Ce(3) with SOC", color = "blue")
plt.plot(temps, soc_cfs_F, label = "Ce(3) with SOC and CFS", color = "green")
plt.xlabel("Temperature (K)")
plt.ylabel("Free Energy (eV)")
plt.title("Relationship Between Free Energy and Temperature")
plt.legend()
plt.savefig("freeenergy.png", dpi=300, bbox_inches="tight")
plt.show()

#Plot Entropy
plt.plot(temps, isolated_entropy, label = "Isolated Ce(3)", color = "red")
plt.plot(temps, soc_entropy, label = "Ce(3) with SOC", color = "blue")
plt.plot(temps, soc_cfs_entropy, label = "Ce(3) with SOC and CFS", color = "green")
plt.xlabel("Temperature (K)")
plt.ylabel("Entropy (eV/K)")
plt.title("Relationship Between Entropy and Temperature")
plt.legend()
plt.savefig("entropy.png", dpi=300, bbox_inches="tight")
plt.show()




