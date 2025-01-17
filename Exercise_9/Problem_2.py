import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

x = np.linspace(0, 20, 1000)  # Vrijednosti x za graf
dfs = [1, 2, 3, 4, 5]  # Stupnjevi slobode

plt.figure(figsize=(8, 6))
for df in dfs:
    cdf = chi2.cdf(x, df)
    plt.plot(x, cdf, label=f"$k={df}$")

plt.title("CDF Chi-Squared distribucije za različite stupnjeve slobode")
plt.xlabel("$x$")
plt.ylabel("$CDF_{\chi^2_k}(x)$")
plt.legend()
plt.grid()
plt.savefig("Chi_square")

x_value = chi2.ppf(0.954, df=1)
print(f"Vrijednost x za koju vrijedi CDF_{{χ²₁}}(x) = 0.954: x = {x_value:.4f}")
