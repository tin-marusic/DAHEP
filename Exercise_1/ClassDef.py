import math
import random

class Boson:
    isFermion = False
    
    def __init__(self, name, spin, momentum):
        self.name = name
        self.spin = spin
        self.momentum = momentum

    def PrintInfo(self):
        print(f"Name: {self.name}")
        print(f"Spin: {self.spin}")
        print(f"Momentum: {self.momentum}")
        print(f"Is Fermion: {self.isFermion} \n")

class Higgs(Boson):
    MassSigma = 1
    
    def __init__(self, name, spin, momentum, MassMean=125):
        super().__init__(name, spin, momentum)
        self.MassMean = MassMean
        self.mass = random.gauss(self.MassMean, Higgs.MassSigma)

    def Energy(self):
        c = 299792458
        momentum_term = (self.momentum * c) ** 2
        mass_term = (self.mass * c ** 2) ** 2
        energy = math.sqrt(momentum_term + mass_term)
        return energy

    def PrintInfo(self):
        print(f"Mass (sampled from Gaussian): {self.mass}")
        super().PrintInfo()
