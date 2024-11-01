import uproot
import matplotlib.pyplot as plt

root = uproot.open("/home/public/data/Lifetime/Lifetime.root")
tree = root.get("Tree;1")
branches = tree.arrays("t")

t_values = [item["t"] for item in branches]

plt.hist(t_values, bins=10, edgecolor='black')  # Možete prilagoditi broj binova (npr. bins=10)
plt.xlabel('Vrijednosti t')
plt.ylabel('Frekvencija')
plt.title('Histogram vrijednosti t')
plt.savefig("Distribucija")