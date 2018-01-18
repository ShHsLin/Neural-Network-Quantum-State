import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
plt.plot(np.genfromtxt(filename))
plt.show()


