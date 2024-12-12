import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Podaci
forces = np.array([1, 2, 3, 4, 5])  # u N
accelerations = np.array([9.8, 21.2, 34.5, 39.9, 48.5])  # u m/s^2
uncertainties = np.array([1.0, 1.9, 3.1, 3.9, 5.1])  # u m/s^2

# Definicija teorijskog modela (Newtonov zakon: F = ma -> a = F / m)
def acceleration_model(force, m):
    return force / m

# Fitovanje podataka
popt, pcov = curve_fit(acceleration_model, forces, accelerations, sigma=uncertainties, absolute_sigma=True)

# Rezultati fitovanja
m_hat = popt[0]  # Procena mase
m_uncertainty = np.sqrt(pcov[0, 0])  # Nesigurnost procene mase

# Grafikon
plt.errorbar(forces, accelerations, yerr=uncertainties, fmt='o', label='Izmereni podaci', capsize=5)
forces_fit = np.linspace(0.5, 5.5, 100)
plt.plot(forces_fit, acceleration_model(forces_fit, m_hat), label=f'Fit: $m = {m_hat:.2f} \pm {m_uncertainty:.2f}$ kg')
plt.xlabel('Sila (N)')
plt.ylabel('Ubrzanje (m/s$^2$)')
plt.title('Fitovanje podataka: Drugi Newtonov zakon')
plt.legend()
plt.grid()
plt.savefig("Problem_1")

# Prikaz rezultata
print(f"Procena mase: m = {m_hat:.2f} kg")
print(f"Nesigurnost mase: ±{m_uncertainty:.2f} kg")