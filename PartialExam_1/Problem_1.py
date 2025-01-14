import uproot
import matplotlib.pyplot as plt
import numpy as np

root = uproot.open("/home/public/data/JPsi/TnPpairs_MC.root")
#print(root.keys())
tree = root.get("Events;1")
#print(tree.keys())
selected_branches = ["ele_pt","ele_eta","ele_phi"]
branches = tree.arrays(selected_branches)

print(branches["ele_pt"])

def calculate_invariant_mass(pt, eta, phi):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2)
    return px, py, pz, E

Data = []

for i in range(10001):
    Px1,Py1,Pz1,E1 = calculate_invariant_mass(branches["ele_pt"][i][0],branches["ele_eta"][i][0],branches["ele_phi"][i][0])
    Px2,Py2,Pz2,E2 = calculate_invariant_mass(branches["ele_pt"][i][1],branches["ele_eta"][i][1],branches["ele_phi"][i][1])

    Data.append(np.sqrt((E1 + E2)**2 - (Px1 + Px2)**2 - (Py1 + Py2)**2 - (Pz1 + Pz2)**2))

    if(i % 1000 == 0):
        print(f"Broj obradenih eventa: {i}")
        
weights = np.ones_like(Data) / len(Data)
plt.hist(Data,bins = 200, color='blue',weights=weights, label="Invariant Mass")


from scipy.stats import norm

def acceptance_rejection_sampling(pdf_func, x_range, max_pdf_value, sample_size=10000):
    """
    Generates random samples based on a given PDF using the acceptance-rejection method.

    Parameters:
        pdf_func (function): The target PDF function to sample from.
        x_range (tuple): The range (min, max) for x values from which to sample.
        max_pdf_value (float): An upper bound for the values of the PDF.
        sample_size (int): The number of samples to generate.

    Returns:
        list: Generated samples following the PDF.
    """
    samples = []
    while len(samples) < sample_size:
        # Draw a candidate x from uniform distribution in the given x range
        x_candidate = np.random.uniform(x_range[0], x_range[1])
        # Draw a uniform y from [0, max_pdf_value]
        y_candidate = np.random.uniform(0, max_pdf_value)
        
        # Accept x_candidate if y_candidate is less than or equal to pdf_func(x_candidate)
        if y_candidate <= pdf_func(x_candidate):
            samples.append(x_candidate)
    
    return samples

def custom_pdf(x, C = 1):
    return C*np.exp(-0.5*x) 

x_range = (0, 10)
max_pdf_value = custom_pdf(0)  #maksimum u x=0

sample_size = 10000
samples = acceptance_rejection_sampling(custom_pdf, x_range, max_pdf_value, sample_size)

weights = np.ones_like(samples) / len(samples)
plt.hist(samples, bins=30, density=True, alpha=0.6, color='skyblue', weights= weights, label="Generated Samples")
x_values = np.linspace(x_range[0], x_range[1], 1000)

distribucija = custom_pdf(x_values,0.503391) # drugi broj je normalizacijski faktor izracunat rjesavanjem integrala od 0 do 10
plt.plot(x_values, distribucija , color="red", label="Target PDF - exponential")
plt.legend()
plt.grid(True)

plt.legend()
plt.xlabel("Invariant Mass (GeV)")
plt.ylabel("Events")
plt.title("Invariant Mass")
plt.savefig("Problem_1")