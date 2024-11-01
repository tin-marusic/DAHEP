import numpy as np
import matplotlib.pyplot as plt
import uproot
from scipy.stats import expon 

root = uproot.open("/home/public/data/Lifetime/Lifetime.root")
tree = root.get("Tree;1")
branches = tree.arrays("t")
t_values = [item["t"] for item in branches]

t_sum = np.sum(t_values)
n_measurements = len(t_values)

def neg2_log_likelihood(tau):
    return 2 * (n_measurements * np.log(tau) + t_sum / tau)

tau_values = np.linspace(1, 1.5, 10000)
neg2_log_likelihood_values = neg2_log_likelihood(tau_values)

min_index = np.argmin(neg2_log_likelihood_values)
theta_hat = tau_values[min_index]
min_neg2_log_likelihood = neg2_log_likelihood_values[min_index]

threshold = min_neg2_log_likelihood + 1

left_side = neg2_log_likelihood_values[:min_index]
left_index = np.argmin(np.abs(left_side - threshold))

# Desna strana
right_side = neg2_log_likelihood_values[min_index:]
right_index = min_index + np.argmin(np.abs(right_side - threshold))

# Vrijednosti tau za interval pouzdanosti
lower_theta = tau_values[left_index]
upper_theta = tau_values[right_index]

plt.plot(tau_values, neg2_log_likelihood_values, label=r'$-2 \ln L(\tau)$')
plt.axhline(y=threshold, color='r', linestyle='--', label=r'$\pm 1$ threshold')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$-2 \ln L(\tau)$')
plt.title(r'Likelihood function for $\tau$ estimation')
plt.legend()
plt.grid(True)
plt.savefig("Log-likehood")

print(f"Minimum je: {theta_hat}")
print(f"Sigma theta minus: {theta_hat - lower_theta}")
print(f"Sigma theta plus: {upper_theta - theta_hat}")

loc, scale = expon.fit(t_values, floc=0)  # 'floc=0' jer eksponencijalna distribucija ima lokaciju 0

# Parametar skale odgovara procjeni za srednje vrijeme života τ
tau_mle = scale
tau_std_mle = tau_mle / np.sqrt(len(t_values))  # Nesigurnost τ prema standardnoj pogrešci

print("Procijenjeno srednje vrijeme života (MLE fit):", tau_mle)
print("Procijenjena nesigurnost za srednje vrijeme života:", tau_std_mle)
