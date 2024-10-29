import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# Parameters for the standard normal distribution
mu = 0
sigma = 1

# Define the interval for x values
x_values = np.linspace(-5, 5, 1000)

# Calculate the Cumulative Density Function (CDF) for each x in the interval
cdf_values = norm.cdf(x_values, mu, sigma)

# Plot the CDF
plt.figure(figsize=(10, 6))
plt.plot(x_values, cdf_values, color="blue", label="CDF of Gauss(µ=0, σ=1)")
plt.title("Cumulative Density Function (CDF) of Standard Normal Distribution")
plt.xlabel("x")
plt.ylabel("Cumulative Probability")
plt.grid(True)
plt.legend()
plt.savefig("CDF")
