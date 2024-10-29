import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def acceptance_rejection_sampling(pdf_func, x_range, max_pdf_value, sample_size=1000):
    """
    Generates random samples based on a given PDF using the acceptance-rejection method.

    Parameters:
        pdf_func (function): The target PDF function to sample from.
        x_range (tuple): The range (min, max) for x values from which to sample.
        max_pdf_value (float): An upper bound for the values of the PDF.
        sample_size (int): The number of samples to generate.

    Returns:
        list: Generated samples following the PDF.
    """
    samples = []
    while len(samples) < sample_size:
        # Draw a candidate x from uniform distribution in the given x range
        x_candidate = np.random.uniform(x_range[0], x_range[1])
        # Draw a uniform y from [0, max_pdf_value]
        y_candidate = np.random.uniform(0, max_pdf_value)
        
        # Accept x_candidate if y_candidate is less than or equal to pdf_func(x_candidate)
        if y_candidate <= pdf_func(x_candidate):
            samples.append(x_candidate)
    
    return samples

# Define a custom PDF function (for example, Gaussian PDF with µ=0, σ=1)
def custom_pdf(x):
    return norm.pdf(x, 0, 1)

# Define the range and max PDF value for sampling
x_range = (-5, 5)
max_pdf_value = custom_pdf(0)  # The peak of the PDF for standard normal is at x=0

# Generate samples
sample_size = 10**4
samples = acceptance_rejection_sampling(custom_pdf, x_range, max_pdf_value, sample_size)

# Plot the histogram of the generated samples
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.6, color='skyblue', label="Generated Samples")
# Overlay the actual PDF for comparison
x_values = np.linspace(x_range[0], x_range[1], 1000)
plt.plot(x_values, custom_pdf(x_values), color="red", label="Target PDF")
plt.title("Samples Generated Using Acceptance-Rejection Method")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig("Random_generator_Gauss")
