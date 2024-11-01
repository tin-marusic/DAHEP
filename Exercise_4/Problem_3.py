import numpy as np
import matplotlib.pyplot as plt

# Vrijednost mjerenog vremena t
t_observed = 1

# Raspon vrijednosti tau
taus = np.linspace(0.1, 5, 100)
likelihoods = (1/taus) * np.exp(-t_observed / taus)

# Crtanje funkcije vjerodostojnosti
plt.plot(taus, likelihoods, label='L(τ)')
plt.xlabel('Vrijednost τ')
plt.ylabel('Funkcija vjerodostojnosti L(τ)')
plt.title('Funkcija vjerodostojnosti za t = 1 s')
plt.legend()
plt.grid(True)
plt.savefig("Likelihood_t=1s")