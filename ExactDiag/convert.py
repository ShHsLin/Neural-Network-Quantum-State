import numpy as np

L = 10
basis = []
for line in open('basisMatrix'+str(L)+'.csv', 'r'):
    basis.append(line[:-1])

newbasis = np.zeros((len(basis), L))
for i in range(len(basis)):
    for j in range(L):
        newbasis[i, j] = basis[i][j]


