import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
# Earray=np.genfromtxt(filename, dtype=str)
# Earray=np.loadtxt(filename).view(complex)
Earray=np.loadtxt(filename, dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})

print(np.sum(Earray))
plt.plot(np.real(Earray))
plt.show()

plt.plot(np.imag(Earray))
plt.show()


