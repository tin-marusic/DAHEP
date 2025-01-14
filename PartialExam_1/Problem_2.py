import uproot
import matplotlib.pyplot as plt
import numpy
from scipy.stats import norm
from scipy.optimize import curve_fit

root = uproot.open("/home/public/data/Gauss.root")
#print(root.keys())
tree = root.get("tree;1")
#print(tree.keys())
selected_branches = ["x_observed"]
branches = tree.arrays(selected_branches)


def gauss(x, C, mu, sigma):
    return C*numpy.exp(-(x-mu)**2/(2.*sigma**2))

x = branches["x_observed"]
y = gauss(x,1, 0, 1) #druga tri broja initial guess 

x = numpy.asarray(x)
x = x.tolist()
yn = y + 0.2 * numpy.random.normal(size=len(x))
yn = numpy.asarray(yn)
yn = yn.tolist()

fig = plt.figure() 
ax = fig.add_subplot(111) 
ax.plot(x, y, c='k', label='Function') 
ax.scatter(x, yn) 
  
# Executing curve_fit on noisy data 
popt, pcov = curve_fit(gauss, x, yn) 
  
#popt returns the best fit values for parameters of the given model (func) 
print (popt) 
  
ym = gauss(x, popt[0], popt[1], popt[2]) 
ax.plot(x, ym, c='r', label='Best fit') 
ax.legend() 
fig.savefig('model_fit.png')

plt.clf()
plt.hist(x, bins=40, density=True, alpha=0.6, color='skyblue',  label="Generated Samples", range= [80,120])
plt.savefig("hist-zad2")