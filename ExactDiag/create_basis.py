import numpy as np
import sys


if __name__ == '__main__':
    basis_list = []
    L = int(sys.argv[1])
    with open("basis_L%d.csv" % L, 'w') as file:
        for idx in range(2**L):
            # basis_list.append(', '.join(bin(idx)[2:].zfill(10)))
            file.write(', '.join(bin(idx)[2:].zfill(L)) + '\n')

    # np.savetxt("basis_L%d.csv" % L, basis_list, delimiter='\n')
