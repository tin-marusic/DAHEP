import uproot
import matplotlib.pyplot as plt
import numpy as np

root = uproot.open("/home/public/data/ggH125/ZZ4lAnalysis.root")
#print(root.keys())
tree = root.get("ZZTree/candTree;1")
#print(tree.keys())
selected_branches = ["LepPt","LepEta","LepPhi"]
branches = tree.arrays(selected_branches)

def calculate_invariant_mass(pt, eta, phi):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2)
    return px, py, pz, E

Data_higgs = []
Data_Z = []

for i in range(len(branches["LepPt"])):
    Px1,Py1,Pz1,E1 = calculate_invariant_mass(branches["LepPt"][i][0],branches["LepEta"][i][0],branches["LepPhi"][i][0])
    Px2,Py2,Pz2,E2 = calculate_invariant_mass(branches["LepPt"][i][1],branches["LepEta"][i][1],branches["LepPhi"][i][1])
    Px3,Py3,Pz3,E3 = calculate_invariant_mass(branches["LepPt"][i][2],branches["LepEta"][i][2],branches["LepPhi"][i][2])
    Px4,Py4,Pz4,E4 = calculate_invariant_mass(branches["LepPt"][i][3],branches["LepEta"][i][3],branches["LepPhi"][i][3])

    Data_Z.append(np.sqrt((E1 + E2)**2 - (Px1 + Px2)**2 - (Py1 + Py2)**2 - (Pz1 + Pz2)**2))

    Data_higgs.append(np.sqrt((E1 + E2 + E3 + E4)**2 - 
                        (Px1 + Px2 + Px3 + Px4)**2 - 
                        (Py1 + Py2 + Py3 + Py4)**2 - 
                        (Pz1 + Pz2 + Pz3 + Pz4)**2))
    if(i % 1000 == 0):
        print(f"Broj obradenih eventa: {i}")


plt.hist(Data_higgs,bins=200,color='red', label="Higgs Mass")
plt.hist(Data_Z,bins = 200, color='blue', label="Z1 Mass")

plt.legend()
plt.xlabel("Invariant Mass (GeV)")
plt.ylabel("Events")
plt.title("Invariant Mass of Higgs and Z1 Bosons")
plt.savefig("Test")