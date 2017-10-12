import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt


def gen_pair(row, V):
    '''
    assume row is an in order array generate a cyclic pairs
    in the row array given with interaction strength V.
    For example: row = [1, 2, 3, 5]
    will gives [(1, 2, V), (2, 3, V), (3, 5, V), (5, 1, V)]
    '''
    return [(row[i], row[(i + 1) % len(row)], V) for i in xrange(len(row))]


def build_H(pairs, L):
    Sx = np.array([[0., 1.],
                   [1., 0.]])
    Sy = np.array([[0., -1j],
                   [1j, 0.]])
    Sz = np.array([[1., 0.],
                   [0., -1.]])

    # S = [Sx, Sy, Sz]
    H = scipy.sparse.csr_matrix((2 ** L, 2 ** L))
    for i, j, V in pairs:
        if i > j:
            i, j = j, i

        print("building", i, j)
        hx = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sx)
        hx = scipy.sparse.kron(hx, scipy.sparse.eye(2 ** (j - i - 1)))
        hx = scipy.sparse.kron(hx, Sx)
        hx = scipy.sparse.kron(hx, scipy.sparse.eye(2 ** (L - j)))

        hy = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sy)
        hy = scipy.sparse.kron(hy, scipy.sparse.eye(2 ** (j - i - 1)))
        hy = scipy.sparse.kron(hy, Sy)
        hy = scipy.sparse.kron(hy, scipy.sparse.eye(2 ** (L - j)))

        hz = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sz)
        hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (j - i - 1)))
        hz = scipy.sparse.kron(hz, Sz)
        hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (L - j)))

        H = H + V * (hx + hy + hz)

    H = scipy.sparse.csr_matrix(H)
    return H


def solve_1d_J1J2(L, J1=1, J2=1):
    lattice = np.arange(L, dtype=int) + 1
    print lattice
    pairs = []
    J1 = J1
    for i in range(1, L + 1):
        pairs = pairs + [(i, (i % L) + 1, J1)]

    J2 = J2
    for i in range(1, L - 1):
        pairs = pairs + [(i, i + 2, J2)]

    pairs += [(L - 1, 1, J2), (L, 2, J2)]

    print('all pairs', pairs)
    H = build_H(pairs, L)

    evals_small, evecs_small = eigsh(H, 6, which='SA')
    print evals_small / L / 4.
    return evals_small, evecs_small


def solve_2d_AFH(Lx, Ly, J=1):
    lattice = np.zeros((Lx, Ly), dtype=int)
    for i in range(Lx):
        for j in range(Ly):
            lattice[i, j] = int(j * Lx + (i+1))

    print lattice
    pairs = []
    for i in range(Lx):
        print lattice[i, :]
        pairs = pairs + gen_pair(lattice[i, :], J)

    for j in range(Ly):
        print lattice[:, j]
        pairs = pairs + gen_pair(lattice[:, j], J)

    print('all pairs', pairs)
    H = build_H(pairs, Lx*Ly)

    evals_small, evecs_small = eigsh(H, 6, which='SA')
    print evals_small / Lx / Ly / 4.
    return evals_small, evecs_small


if __name__ == "__main__":
    import sys
    model = sys.argv[1]
    if model == '1dJ1J2':
        L, J1, J2 = sys.argv[2:]
        L, J1, J2 = int(L), float(J1), float(J2)
        print("python 1dJ1J2 L=%d J1=%d J2=%d" % (L, J1, J2) )
        evals_small, evecs_small = solve_1d_J1J2(L, J1, J2)
    elif model == '2dAFH':
        evals_small, evecs_small = solve_2d_AFH(4, 4)
    else:
        print("error in input arguments:\ncurrently support for 1dJ1J2, 2dAFH")
        raise NotImplementedError

# plt.plot(np.real(evecs_small[:, 0]), label='real')
# plt.plot(np.imag(evecs_small[:, 0]), label='imag')
# plt.legend()
# plt.show()
