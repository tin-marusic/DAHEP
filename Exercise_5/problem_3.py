import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Podaci
forces = np.array([1, 2, 3, 4, 5])  # Sile u N
accelerations = np.array([9.8, 21.2, 34.5, 39.9, 48.5])  # Ubrzanja u m/s^2
uncertainties = np.array([1.0, 1.9, 3.1, 3.9, 5.1])  # Nesigurnosti ubrzanja u m/s^2

# Definicija teorijskog modela (Newtonov zakon: F = ma -> a = F / m)
def acceleration_model(force, m):
    """
    Model ubrzanja na osnovu druge Newtonove zakonitosti.
    Args:
        force (array-like): Primijenjene sile.
        m (float): Masa objekta (parametar za procjenu).
    Returns:
        array-like: Ubrzanja koja odgovaraju silama za zadanu masu m.
    """
    return force / m

# Fitovanje podataka koristeći funkciju ubrzanja i nesigurnosti
popt, pcov = curve_fit(acceleration_model, forces, accelerations, sigma=uncertainties, absolute_sigma=True)

# Rezultati fitovanja
m_hat = popt[0]  # Procjena mase (najbolja vrijednost)
m_uncertainty = np.sqrt(pcov[0, 0])  # Nesigurnost procjene mase (standardna devijacija)

# Računanje χ2 funkcije za različite vrijednosti parametra m
m_values = np.linspace(m_hat - 3 * m_uncertainty, m_hat + 3 * m_uncertainty, 100)  # Raspon za iscrtavanje χ2
chi2_values = []
for m in m_values:
    residuals = (accelerations - acceleration_model(forces, m)) / uncertainties  # Standardizirane rezidue
    chi2 = np.sum(residuals**2)  # Suma kvadrata rezidua
    chi2_values.append(chi2)

# Grafikon χ2 funkcije
plt.plot(m_values, chi2_values, label="$\\chi^2$ funkcija")
plt.axvline(m_hat, color='red', linestyle='--', label=f'Procjena $m = {m_hat:.2f}$')  # Vertikalna linija za m_hat
plt.axvline(m_hat - m_uncertainty, color='blue', linestyle='--', label=f'$m - \\sigma_m = {m_hat - m_uncertainty:.2f}$')  # Linija za m_hat - sigma
plt.axvline(m_hat + m_uncertainty, color='blue', linestyle='--', label=f'$m + \\sigma_m = {m_hat + m_uncertainty:.2f}$')  # Linija za m_hat + sigma
plt.axhline(min(chi2_values) + 1, color='green', linestyle='--', label='Granica $\\Delta\\chi^2 = 1$')  # Horizontalna linija za χ2 + 1
plt.xlabel('Masa $m$ (kg)')
plt.ylabel('$\\chi^2$')
plt.title('Grafikon $\\chi^2$ funkcije')
plt.legend()
plt.grid()
plt.savefig('chi2_plot_with_uncertainties.png')  # Spremanje grafikona u datoteku
plt.show()

# Prikaz rezultata
print(f"Procjena mase: m = {m_hat:.2f} kg")
print(f"Nesigurnost mase: ±{m_uncertainty:.2f} kg")

# Dodatno objašnjenje:
# Procjena parametra m dolazi iz minimuma $\\chi^2$ funkcije. Nesigurnost mase ($\\sigma_m$) određena je intervalom gdje $\\chi^2$ raste za 1 od minimalne vrijednosti.
# Linije na grafikonu prikazuju procjenu mase (crvena), granice nesigurnosti (plave), i horizontalnu granicu za $\\Delta\\chi^2 = 1$."
