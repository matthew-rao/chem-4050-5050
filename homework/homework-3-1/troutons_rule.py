# Trouton's Rule, Regression, and Uncertainty Analysis

#Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

#Import ols_slope, ols_intercept, and ols functions from Chapter 7

def ols_slope(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator

def ols_intercept(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = ols_slope(x, y)
    return y_mean - slope * x_mean

def ols(x, y):
    slope = ols_slope(x, y)
    intercept = ols_intercept(x, y)
    return slope, intercept

#Import CSV file
df = pd.read_csv("trouton.csv")

#Sort CSV file and extract Tb and Hv values
df_Tb = df[["Class", "T_B (K)"]]
df_Hv = df[["Class", "H_v (kcal/mol)"]]

#Convert Hv units (kcal/mol to J/mol-k)
df_Hv_modified = df_Hv * 4184

#Define slope and intercept values, create linear fit
entropy_vap, intercept = ols(df_Tb["T_B (K)"], df_Hv_modified["H_v (kcal/mol)"])
linear_fit = entropy_vap * df_Tb["T_B (K)"] + intercept

#Calculate Residuals
residuals = df_Hv_modified["H_v (kcal/mol)"] - linear_fit

#Calculate SSE
sse = np.sum(residuals ** 2)

#Calculate variance
variance = sse / (len(residuals) - 2)

#SE of slope and intercept
def se_slope(x):
    numerator = variance
    x_mean = np.mean(x)
    denominator = np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)

slope_se = se_slope(df_Tb["T_B (K)"])

def se_intercept(x):
    numerator = variance
    x_mean = np.mean(x)
    denominator = len(x) * np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)

intercept_se = se_intercept(df_Tb["T_B (K)"])

#Calculate 95% Confidence Intervals for both slope and intercept
def confidence_interval(x, se, residuals, confidence_level):
    # Calculate the critical t-value
    n_data_points = len(x)
    df = n_data_points - 2  # degrees of freedom
    alpha = 1 - confidence_level
    critical_t_value = t.ppf(1 - alpha/2, df)

    # Calculate the confidence interval
    return critical_t_value * se

slope_ci = confidence_interval(df_Tb["T_B (K)"], slope_se, residuals, confidence_level = 0.95)
intercept_ci = confidence_interval(df_Tb["T_B (K)"], intercept_se, residuals, confidence_level = 0.95)

print(f"slope: {entropy_vap:.3f} +/- {slope_ci:.3f}")
print(f"intercept: {intercept:.3f} +/- {intercept_ci:.3f}")

#Plot graph of Trouton's Rule and save as PNG
plt.plot(df_Tb["T_B (K)"], linear_fit, label = "Linear Fit")
df_plot = df[["Class", "T_B (K)"]].copy()
df_plot["H_v (J/mol-k)"] = df["H_v (kcal/mol)"] * 4184.0
df_plot.head()

for class_name, group in df_plot.groupby("Class"):
    plt.scatter(
        group["T_B (K)"],            # x-values
        group["H_v (J/mol-k)"],      # y-values
        label=class_name,
        alpha = 0.8
    )
plt.xlabel("Boiling Point $T_B$ (K)")
plt.ylabel("Enthalpy of vaporization $H_v$ (J/mol*k)")
plt.title("Trouton's Rule: Relationship Between $H_v$ and $T_B$")
ax = plt.gca()
ax.text(
    0.98, 0.02,
    r"$H_v = a \cdot T_B + b$" "\n"
    r"$a = 103.855 \pm 6.396\ J/mol*k$" "\n"
    r"$b = -4844.603 \pm 1.306\ J/mol$",
    transform=ax.transAxes, ha="right", va="bottom",
    fontsize=12, zorder=5,
    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85)
)
plt.legend()
plt.savefig("trouton.png", dpi=300, bbox_inches="tight")
plt.show()

#Comment on Trouton's Rule
#According to Trouton's Rule, the entropy of vaporization is 88 J/mol*K, which is off from our predicted 103.855 value
#Looking at the graph, we see that the mean is less robust to extreme values
#That is, Trouton's Rule does NOT hold well for metals (those with high BPs)
#However, it can be said that Trouton's rule does hold well for liquids and those with lower BPs.
#Running this experiment again with just liquids and excluding the solid values would likely decrease the gap between the observed and predicted



