#Import necessary packages
import matplotlib.pyplot as plt
import pandas as pd

#Import csv files
df_iso = pd.read_csv("work_volume_iso.csv")
df_adi = pd.read_csv("work_volume_adi.csv")

#Plot graph of work done (y) vs final volume (x) for both processes
plt.plot(df_iso["Volume (m^3)"], df_iso["Work (J)"], color = "blue", label = "Isothermal process")
plt.plot(df_adi["Volume (m^3)"], df_adi["Work (J)"], color = "red", label = "Adiabatic process")
plt.xlabel("Final Volume of Ideal Gas (m^3)")
plt.ylabel("Work Done on Ideal Gas (J)")
plt.title("Relationship Between Work and Final Volume of Ideal Gas")
plt.legend()
plt.savefig("work.png", dpi=300, bbox_inches="tight")
plt.show()
