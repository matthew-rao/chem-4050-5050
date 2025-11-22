#Adiabatic process

#Import necessary packages
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd

#Initialize variable
work_done = []

#Define function
def compute_work_adi(V_i, V_f, R = 8.314, n = 1, T = 300, gamma = 1.4):
    """
    Computes the work done on an adiabatic expansion from Vi to Vf

    Parameters:
    V_i (int or float): the initial value of the volume of the ideal gas
    V_f (int or float): the final value of the volume of the ideal gas
    n (int or float): the moles of our ideal gas
    T (int or float): temperature in kelvin

    Returns:
    List: the work done on the adiabatic system in J, across an evenly spaced V_i to V_f
    """

    #Calculate constant (holds true at V_i and V_f)
    pressure = (n * R * T) / V_i
    constant = pressure * (V_i ** gamma)

    #Perform Integration
    volumes = np.linspace(V_i, V_f, 1000)
    integrand = constant / (volumes ** gamma)
    integral = -trapezoid(integrand, volumes)

    return integral

#Find the work done from V_i to 3V_i
V_i = 0.1
for volume in np.linspace(V_i, 3 * V_i, 1000):
    work = compute_work_adi(V_i, volume, R = 8.314, n = 1, T = 300, gamma = 1.4)
    work_done.append(work)

#Combine arrays
volumes = np.linspace(V_i, V_i * 3, 1000)
combined_data = np.column_stack((work_done, volumes))

#Convert arrays to dataframe and CSV file
work_volume_df = pd.DataFrame(combined_data)
work_volume_df.to_csv("work_volume_adi.csv", header = ["Work (J)", "Volume (m^3)"], index = False)
    