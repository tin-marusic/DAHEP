import numpy as np
import matplotlib.pyplot as plt

# Values for t
t = np.linspace(0, 10, 1000)

# Different values of tau
taus = [0.5, 1, 2, 5]

# Plotting the PDFs for different tau values
plt.figure(figsize=(10, 6))
for tau in taus:
    pdf = (1/tau) * np.exp(-t/tau)
    plt.plot(t, pdf, label=f'tau = {tau}')

plt.xlabel('t')
plt.ylabel('f(t; τ)')
plt.title('Exponential PDF for Different Values of τ')
plt.legend()
plt.grid(True)
plt.savefig("theoretical_PDF")

t_val = 1
tau_val = 2
probability = 1 - np.exp(-t_val / tau_val)
print(f"Vjerojatnost mjerenja t <= 1s: {probability*100}%")
