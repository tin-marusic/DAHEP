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

# Računanje χ2 funkcije za različite vrijednosti parametra m
m_values = np.linspace(m_hat - 3 * m_uncertainty, m_hat + 3 * m_uncertainty, 100)
chi2_values = []
for m in m_values:
    residuals = (accelerations - acceleration_model(forces, m)) / uncertainties
    chi2 = np.sum(residuals**2)
    chi2_values.append(chi2)

# Grafikon χ2 funkcije
plt.plot(m_values, chi2_values, label="$\chi^2$ funkcija")
plt.axvline(m_hat, color='red', linestyle='--', label=f'Procjena $m = {m_hat:.2f}$')
plt.axhline(min(chi2_values) + 1, color='green', linestyle='--', label='Granica $\Delta\chi^2 = 1$')
plt.xlabel('Masa $m$ (kg)')
plt.ylabel('$\chi^2$')
plt.title('Grafikon $\chi^2$ funkcije')
plt.legend()
plt.grid()
plt.savefig('chi2_plot.png')  # Spremanje grafikona u datoteku
plt.show()

# Prikaz rezultata
print(f"Procena mase: m = {m_hat:.2f} kg")
print(f"Nesigurnost mase: ±{m_uncertainty:.2f} kg")

# Dodatno objašnjenje:
# Procjena parametra m dolazi iz minimuma $\chi^2$ funkcije. Nesigurnost mase ($\sigma_m$) određena je intervalom gdje $\chi^2$ raste za 1 od minimalne vrijednosti. Ova metoda omogućava direktno poređenje s prethodnim rezultatima dobivenim pomoću fitovanja."
