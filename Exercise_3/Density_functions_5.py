import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from Density_functions_3 import acceptance_rejection_sampling
import time

# Parameters for the standard normal distribution
mu = 0
sigma = 1

sample_size = 10**4

# Inversion method: generate random numbers based on the CDF
def inversion_method(mu, sigma, sample_size):
    u_values = np.random.uniform(0, 1, sample_size)
    x_values = norm.ppf(u_values, mu, sigma)  # Inverse of CDF
    return x_values

# Generate samples using the inversion method
start_time_inv = time.time()
inversion_samples = inversion_method(mu, sigma, sample_size)
end_time_inv = time.time()

# Target PDF for standard normal
def normal_pdf(x):
    return norm.pdf(x, mu, sigma)

x_range = (-5, 5)
max_pdf_value = normal_pdf(0)
start_time_ar = time.time()
ar_samples = acceptance_rejection_sampling(normal_pdf, x_range, max_pdf_value, sample_size)
end_time_ar = time.time()

# Plot the histogram of both methods
plt.figure(figsize=(12, 6))
plt.hist(inversion_samples, bins=30, density=True, alpha=0.5, label="Inversion Method Samples", color="blue")
plt.hist(ar_samples, bins=30, density=True, alpha=0.5, label="Acceptance-Rejection Samples", color="orange")
x_values = np.linspace(-5, 5, 1000)
plt.plot(x_values, normal_pdf(x_values), color="red", label="Target PDF")
plt.title("Comparison of Inversion Method and Acceptance-Rejection Method")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig("comparison of inverse and acceptance-rejection methods")

# Efficiency comparison
time_inversion = end_time_inv - start_time_inv
time_acceptance_rejection = end_time_ar - start_time_ar

print("Inversion Method: Time taken =", time_inversion)
print("Acceptance-Rejection Method: Time taken =", time_acceptance_rejection)
