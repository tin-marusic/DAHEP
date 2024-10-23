from ClassDef import Boson, Higgs

boson = Boson(name="Photon", spin=1, momentum=10)
print("Boson Information:")
boson.PrintInfo()

higgs = Higgs(name="Higgs", spin=0, momentum=20)
print("Higgs Information:")
higgs.PrintInfo()

energy = higgs.Energy()
print(f"\nHiggs Boson Energy: {energy}")
