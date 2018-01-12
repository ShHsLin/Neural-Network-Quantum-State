'''
The Ground State energy of Antiferromagnet Heisenberg model
from Bethe Ansatz is given here.

For derivation see
https://arxiv.org/pdf/cond-mat/9809163.pdf
'''
import numpy as np

N = 256
r = int(N/2)
I = np.arange(-N+2,N-2+1,4)/4.
z = np.zeros(r)
new_z = np.zeros(r)
num_iter = 200
E_GS = 0.
for iter_idx in range(num_iter):
    for i in range(r):
        sum_phi = np.sum( 2. * np.arctan((z[i] - z)/2.))
        new_z[i] = np.tan(np.pi / N * I[i] + sum_phi /(2.*N) )
    z = new_z
    new_E_GS = np.sum( -2/(1.+(z ** 2)) )
    if (np.abs((new_E_GS-E_GS)/new_E_GS) > 1e-14):
        E_GS = new_E_GS
        print(iter_idx, E_GS/N)
    else:
        break

print("GS energy for %d-site AFH : %.16f" % (N,E_GS/N+0.25))
