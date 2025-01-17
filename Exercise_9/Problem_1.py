from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

num_experiments = 10000
sample_size = 1000
true_mu = 5
true_sigma = 2
likelihood_ratios = []

for _ in range(num_experiments):

    data = np.random.normal(loc=true_mu, scale=true_sigma, size=sample_size)
    
    sigma_h0 = np.std(data, ddof=1)  
    log_likelihood_h0 = -0.5 * sample_size * np.log(2 * np.pi * sigma_h0**2) \
                        - np.sum((data - true_mu)**2) / (2 * sigma_h0**2)

    mu_h1 = np.mean(data)
    sigma_h1 = np.std(data, ddof=1)
    log_likelihood_h1 = -0.5 * sample_size * np.log(2 * np.pi * sigma_h1**2) \
                        - np.sum((data - mu_h1)**2) / (2 * sigma_h1**2)

    likelihood_ratio = -2 * (log_likelihood_h0 - log_likelihood_h1)
    likelihood_ratios.append(likelihood_ratio)
    
    
plt.hist(likelihood_ratios, bins=50, density=True, alpha=0.7, label="Simulacija")

x = np.linspace(0, 10, 1000)
plt.plot(x, chi2.pdf(x, df=1), 'r-', label="$\\chi^2_1$ distribucija")

plt.title("Wilkova teorema: Likelihood-ratio vs $\\chi^2_1$")
plt.xlabel("Likelihood-ratio test statistika")
plt.ylabel("Gustoća")
plt.legend()
plt.grid()
plt.savefig("Wilks_normal")

