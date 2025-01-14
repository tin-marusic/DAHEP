import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Number of simulations and sample size
num_sums = 10000
sample_size = 100

# Initialize subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

def plot_distribution(ax, data, clt_mean, clt_std, title):
    # Plot histogram of the data
    ax.hist(data, bins=50, density=True, alpha=0.6, color='blue', label='Simulated')
    
    # Generate the CLT predicted Gaussian curve
    x = np.linspace(min(data), max(data), 1000)
    y = norm.pdf(x, loc=clt_mean, scale=clt_std)
    ax.plot(x, y, color='red', label='CLT Prediction')
    
    # Add labels and legend
    ax.set_title(title)
    ax.set_xlabel('Sum')
    ax.set_ylabel('Density')
    ax.legend()

# Exponential distribution parameters
beta = 0.1
exponential_samples = np.sum(np.random.exponential(scale=beta, size=(num_sums, sample_size)), axis=1)
clt_mean_exp = sample_size * beta
clt_std_exp = np.sqrt(sample_size * beta**2)
plot_distribution(axes[0], exponential_samples, clt_mean_exp, clt_std_exp, "Exponential Distribution (CLT in Action)")

# Gaussian distribution parameters
mu_values = np.linspace(0, 100, num=101)
sigma = 5
gaussian_samples = np.sum(np.random.normal(loc=mu_values.mean(), scale=sigma, size=(num_sums, sample_size)), axis=1)
clt_mean_gauss = sample_size * mu_values.mean()
clt_std_gauss = np.sqrt(sample_size * sigma**2)
plot_distribution(axes[1], gaussian_samples, clt_mean_gauss, clt_std_gauss, "Gaussian Distribution (CLT in Action)")

# Poisson distribution parameters
lambda_values = np.arange(1, 101)
lambda_mean = lambda_values.mean()
poisson_samples = np.sum(np.random.poisson(lam=lambda_mean, size=(num_sums, sample_size)), axis=1)
clt_mean_pois = sample_size * lambda_mean
clt_std_pois = np.sqrt(sample_size * lambda_mean)
plot_distribution(axes[2], poisson_samples, clt_mean_pois, clt_std_pois, "Poisson Distribution (CLT in Action)")

# Adjust layout and show
plt.tight_layout()
plt.savefig("CLT")