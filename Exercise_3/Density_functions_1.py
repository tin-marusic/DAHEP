import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define three different sets of mean (µ) and standard deviation (σ) for Gaussian distributions
params = [(0, 1), (200, 10), (200, 2)]
x_values = np.linspace(-50, 300, 1000)

# Plot Gaussian distributions for the three sets of parameters
plt.figure(figsize=(12, 6))
for mu, sigma in params:
    y_values = norm.pdf(x_values, mu, sigma) #pdf - Probability Density Function
    plt.plot(x_values, y_values, label=f"µ = {mu}, σ = {sigma}")

plt.title("Gaussian Distributions with Different Parameters")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.savefig("Gaussian")

# Probability Calculations for Gauss(µ = 200GeV, σ = 2GeV)
mu_particle = 200  # mean
sigma_particle = 2  # standard deviation

# 1. Probability of mass 205 GeV or more
prob_205_or_more = 1 - norm.cdf(205, mu_particle, sigma_particle) #cdf - Cumulative Distribution Function

# 2. Probability of mass between 199 and 201 GeV
prob_199_to_201 = norm.cdf(201, mu_particle, sigma_particle) - norm.cdf(199, mu_particle, sigma_particle)

# 3. Probability of independently producing two particles with masses above 203 GeV
prob_above_203 = 1 - norm.cdf(203, mu_particle, sigma_particle)
prob_two_particles_above_203 = prob_above_203 ** 2

print(f"probability to produce the new particle with a mass of 205 GeV or more: {prob_205_or_more*100}%")
print(f"probability to produce the new particle with a mass between 199 and 201 GeV: {prob_199_to_201*100}%")
print(f"probability to independently produce two new particles with masses above 203 GeV: { prob_two_particles_above_203*100}%")