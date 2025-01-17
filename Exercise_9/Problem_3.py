from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
import uproot

root = uproot.open("/home/public/data/GaussData.root")
tree = root.get("tree;1")
print(tree.keys())
selected_branches = ["x_observed"]
branches = tree.arrays(selected_branches)

data = branches["x_observed"]
mu_0 = np.linspace(4.5,5.5,1000)
sigma_h0 = np.std(data, ddof=1)  
likelihood_ratios = []

for mu in mu_0:

    log_likelihood_h0 = -0.5 * len(data) * np.log(2 * np.pi * sigma_h0**2) \
                            - np.sum((data - mu)**2) / (2 * sigma_h0**2)

    mu_h1 = np.mean(data)
    sigma_h1 = np.std(data, ddof=1)
    log_likelihood_h1 = -0.5 * len(data) * np.log(2 * np.pi * sigma_h1**2) \
                            - np.sum((data - mu_h1)**2) / (2 * sigma_h1**2)

    likelihood_ratio = -2 * (log_likelihood_h0 - log_likelihood_h1)
    likelihood_ratios.append(likelihood_ratio)
    
    
log_likelihood_ratios = np.array(likelihood_ratios)

# Pronađi granice (95.4% interval)
threshold = chi2.ppf(0.954, df=1)
bounds = mu_0[log_likelihood_ratios <= threshold]
lower_bound, upper_bound = bounds[0], bounds[-1]

# Plot Likelihood Ratio kao funkcija μ
plt.figure(figsize=(10, 6))
plt.plot(mu_0, log_likelihood_ratios, label="Likelihood Ratio", color="blue")
plt.axhline(y=threshold, color="red", linestyle="--", label=f"95.4% Threshold ({threshold:.2f})")
plt.axvline(x=lower_bound, color="green", linestyle="--", label=f"Lower Bound ({lower_bound:.2f})")
plt.axvline(x=upper_bound, color="green", linestyle="--", label=f"Upper Bound ({upper_bound:.2f})")

# Postavke grafa
plt.title("Likelihood Ratio kao funkcija parametra $\mu$")
plt.xlabel("$\mu$")
plt.ylabel("Likelihood Ratio")
plt.legend()
plt.grid()
plt.savefig("problem_3")