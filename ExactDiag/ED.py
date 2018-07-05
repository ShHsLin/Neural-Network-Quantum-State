import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt


def gen_pair(row, V, PBC=True):
    '''
    assume row is an in order array generate a cyclic pairs
    in the row array given with interaction strength V.
    For example: row = [1, 2, 3, 5]
    will gives [(1, 2, V), (2, 3, V), (3, 5, V), (5, 1, V)]
    '''
    if PBC == True:
        return [(row[i], row[(i + 1) % len(row)], V) for i in range(len(row))]
    else:
        return [(row[i], row[(i + 1) % len(row)], V) for i in range(len(row)-1)]

def gen_pair_2d_nnn(plaquette, V):
    '''
    assume the plaquette indices are given in cyclic order,
    return two pair of the cross nnn term interaction.
    For example: plaquette = [  1, 2,
                                5, 6]
    wiil gives[(1, 6, V), (2, 5, V)]
    '''
    return [(plaquette[0], plaquette[2], V), (plaquette[1], plaquette[3], V)]

def build_H_one_body(sites, L, H=None, sx=True, sy=True, sz=True):
    Sx = np.array([[0., 1.],
                   [1., 0.]])
    Sy = np.array([[0., -1j],
                   [1j, 0.]])
    Sz = np.array([[1., 0.],
                   [0., -1.]])

    # S = [Sx, Sy, Sz]
    if H is None:
        H = scipy.sparse.csr_matrix((2 ** L, 2 ** L))
    else:
        pass

    for i, V in sites:
        print("building", i)
        if sx:
            hx = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sx)
            hx = scipy.sparse.kron(hx, scipy.sparse.eye(2 ** (L - i)))
            H = H + V * hx

        if sy:
            hy = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sy)
            hy = scipy.sparse.kron(hy, scipy.sparse.eye(2 ** (L - i)))
            H = H + V * hy

        if sz:
            hz = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sz)
            hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (L - i)))
            H = H + V * hz

    H = scipy.sparse.csr_matrix(H)
    return H

def build_H_two_body(pairs, L, H=None, sxsx=True,
                     sysy=True, szsz=True):
    Sx = np.array([[0., 1.],
                   [1., 0.]])
    Sy = np.array([[0., -1j],
                   [1j, 0.]])
    Sz = np.array([[1., 0.],
                   [0., -1.]])

    # S = [Sx, Sy, Sz]
    if H is None:
        H = scipy.sparse.csr_matrix((2 ** L, 2 ** L))
    else:
        pass

    for i, j, V in pairs:
        if i > j:
            i, j = j, i

        print("building", i, j)
        if sxsx:
            hx = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sx)
            hx = scipy.sparse.kron(hx, scipy.sparse.eye(2 ** (j - i - 1)))
            hx = scipy.sparse.kron(hx, Sx)
            hx = scipy.sparse.kron(hx, scipy.sparse.eye(2 ** (L - j)))
            H = H + V * hx

        if sysy:
            hy = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sy)
            hy = scipy.sparse.kron(hy, scipy.sparse.eye(2 ** (j - i - 1)))
            hy = scipy.sparse.kron(hy, Sy)
            hy = scipy.sparse.kron(hy, scipy.sparse.eye(2 ** (L - j)))
            H = H + V * hy

        if szsz:
            hz = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sz)
            hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (j - i - 1)))
            hz = scipy.sparse.kron(hz, Sz)
            hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (L - j)))
            H = H + V * hz

    H = scipy.sparse.csr_matrix(H)
    return H

def build_Sx(L):
    Sx = np.array([[0., 1.],
                   [1., 0.]])
    hx = scipy.sparse.csr_matrix(Sx)
    for i in range(1,L):
        print(i)
        hx = scipy.sparse.kron(hx, Sx)

    return hx

def spin_spin_correlation(site_i, site_j, L, vector):
    Sz = np.array([[1., 0.],
                   [0., -1.]])

    print("correlation between", site_i, site_j)
    hz = scipy.sparse.kron(scipy.sparse.eye(2 ** (site_i - 1)), Sz)
    hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (site_j - site_i - 1)))
    hz = scipy.sparse.kron(hz, Sz)
    hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (L - site_j)))
    SzSz = scipy.sparse.csr_matrix(hz)
    return vector.conjugate().dot(SzSz.dot(vector))

def sz_expectation(site_i, vector, L):
    Sz = np.array([[1., 0.],
                   [0., -1.]])

    print("spin z expectation value on site %d" % site_i)
    hz = scipy.sparse.kron(scipy.sparse.eye(2 ** (site_i - 1)), Sz)
    hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (L - site_i)))
    Sz = scipy.sparse.csr_matrix(hz)
    return vector.conjugate().dot(Sz.dot(vector))

def gen_H_1d_J1J2(L, J1=1, J2=0.):
    lattice = np.arange(L, dtype=int) + 1
    print(lattice)
    pairs = []
    J1 = J1
    for i in range(1, L + 1):
        pairs = pairs + [(i, (i % L) + 1, J1)]

    J2 = J2
    for i in range(1, L - 1):
        pairs = pairs + [(i, i + 2, J2)]

    pairs += [(L - 1, 1, J2), (L, 2, J2)]

    print('all pairs', pairs)
    H = build_H_two_body(pairs, L)
    return H

def gen_H_1d_Ising(L, J, g=0, h=0, PBC=False):
    # Solving -J szsz + g sx - h sz
    lattice = np.arange(L, dtype=int) + 1
    print(lattice)
    sx_sites = [(i, g) for i in range(1, L+1)]
    sz_sites = [(i, -h) for i in range(1, L+1)]
    szsz_pairs = []
    if PBC:
        for i in range(1, L + 1):
            szsz_pairs = szsz_pairs + [(i, (i % L) + 1, -J)]
    else:
        for i in range(1, L):
            szsz_pairs = szsz_pairs + [(i, (i % L) + 1, -J)]

    print('all pairs', szsz_pairs)
    H = build_H_two_body(szsz_pairs, L, sxsx=False, sysy=False)
    H = build_H_one_body(sx_sites, L, H=H, sy=False, sz=False)
    H = build_H_one_body(sz_sites, L, H=H, sx=False, sy=False)
    return H

def gen_local_H_1d_Ising(L, J, g=0, h=0, PBC=False):
    # Solving -1/2(J szsz + J szsz) + g sx - h sz
    H_list = []
    lattice = np.arange(L, dtype=int) + 1
    print(lattice)
    sx_sites = [[(i, g)] for i in range(1, L+1)]
    sz_sites = [[(i, -h)] for i in range(1, L+1)]
    szsz_pairs = [None] * L
    if PBC:
        raise NotImplementedError
    else:
        szsz_pairs[0] = [(1,2,-J/2)]
        szsz_pairs[L-1] = [(L-1, L, -J/2)]
        for i in range(1, L-1):
            szsz_pairs[i] = [(i, i+1, -J/2), (i+1, i+2, -J/2)]

    print('all pairs', szsz_pairs)
    for i in range(L):
        H = build_H_two_body(szsz_pairs[i], L,H=None, sxsx=False, sysy=False)
        H = build_H_one_body(sx_sites[i], L, H=H, sy=False, sz=False)
        H = build_H_one_body(sz_sites[i], L, H=H, sx=False, sy=False)
        H_list.append(np.array(H.todense()))

    return H_list

def measure_local_H(psi, H_list):
    L = len(H_list)
    H_local = np.zeros(L, dtype=np.complex)
    for i in range(L):
        H_local[i] = psi.conj().T.dot(H_list[i]).dot(psi)

    return H_local

def gen_H_2d_J1J2(Lx, Ly, J1=1, J2=0.):
    lattice = np.zeros((Lx, Ly), dtype=int)
    for i in range(Lx):
        for j in range(Ly):
            lattice[i, j] = int(j * Lx + (i+1))

    print(lattice)
    pairs = []
    # NN interaction : J1
    for i in range(Lx):
        print(lattice[i, :])
        pairs = pairs + gen_pair(lattice[i, :], J1)

    for j in range(Ly):
        print(lattice[:, j])
        pairs = pairs + gen_pair(lattice[:, j], J1)

    # NNN interaction : J2
    for i in range(Lx):
        for j in range(Ly):
            plaquette=[lattice[i,j]]
            plaquette.append(lattice[(i+1)%Lx, j])
            plaquette.append(lattice[(i+1)%Lx, (j+1)%Ly])
            plaquette.append(lattice[i, (j+1)%Ly])
            pairs = pairs + gen_pair_2d_nnn(plaquette, J2)

    print('all pairs', pairs)
    global H
    H = build_H_two_body(pairs, Lx*Ly)
    return H

def check_phase(vector, dim=1, site=None):
    '''
    check the phase of one site translation
    '''
    if dim==1:
        new_vector = np.zeros_like(vector)
        len_v = new_vector.size
        new_vector[:int(len_v/2)] = vector[::2]
        new_vector[int(len_v/2):] = vector[1::2]
        return new_vector.conjugate().dot(vector)
    if dim==2:
        new_vector = np.copy(vector)
        len_v = new_vector.size
        for i in range(site):
            temp_vec = np.zeros_like(new_vector)
            temp_vec[:int(len_v/2)] = new_vector[::2]
            temp_vec[int(len_v/2):] = new_vector[1::2]
            new_vector = np.copy(temp_vec)
        return new_vector.conjugate().dot(vector)

def store_eig_vec(evals_small, evecs_small, filename):
    idx_min = np.argmin(evals_small)
    print("GS energy: %f" % evals_small[idx_min])
    vec_r = np.real(evecs_small[:,idx_min])
    vec_i = np.imag(evecs_small[:,idx_min])
    vec_r = vec_r / np.linalg.norm(vec_r)
    vec_i = vec_i / np.linalg.norm(vec_i)
    if np.abs(vec_r.dot(vec_i)) - 1. < 1e-6:
        print("Eigen Vec can be casted as real")
        log_file = open(filename, 'wb')
        np.savetxt(log_file, vec_r, fmt='%.8e', delimiter=',')
        log_file.close()
    else:
        print(np.abs(vec_r.dot(vec_i)) - 1.)
        print("Complex Eigen Vec !!!")
        print("The real part <E> : %f " %  vec_r.T.dot(H.dot(vec_r)) )
        print("The imag part <E> : %f " %  vec_i.T.dot(H.dot(vec_i)) )

    return


if __name__ == "__main__":
    import sys
    model = sys.argv[1]
    if model == '1dJ1J2':
        L, J1, J2 = sys.argv[2:]
        L, J1, J2 = int(L), float(J1), float(J2)
        N = L
        print("python 1dJ1J2 L=%d J1=%f J2=%f" % (L, J1, J2) )
        H = gen_1d_J1J2(L, J1, J2)
        evals_small, evecs_small = eigsh(H, 6, which='SA')
        print(evals_small / L / 4.)

        eig_filename = 'EigVec/ES_%s_L%d_J2_%d.csv' % (model[:2], L, J2*10)
        store_eig_vec(evals_small, evecs_small, eig_filename)
        print("check one site translation phase : {:.2f}".format(check_phase(evecs_small[:,0])))
    elif model == '2dJ1J2':
        Lx, Ly, J1, J2 = sys.argv[2:]
        Lx, Ly, J1, J2 = int(Lx), int(Ly), float(J1), float(J2)
        N = Lx * Ly
        H = gen_H_2d_J1J2(Lx, Ly, J1, J2)
        evals_small, evecs_small = eigsh(H, 6, which='SA')
        print('Energy : ', evals_small / Lx / Ly / 4.)
        eig_filename = 'EigVec/ES_%s_L%dx%d_J2_%d.csv' % (model[:2], Lx, Ly, J2*10)
        store_eig_vec(evals_small, evecs_small, eig_filename)
        print("check n={0:d} site translation phase : {1:.2f}".format(Lx, check_phase(evecs_small[:,0], 2, Lx)))
    elif model == '1dIsing':
        L, J, g, h = sys.argv[2:]
        # Solving -J szsz + g sx + h sz
        L, J, g, h = int(L), float(J), float(g), float(h)
        N = L
        print("python 1dIsing L=%d, J=%f, g=%f, h=%f" % (L, J, g, h))
        H = gen_H_1d_Ising(L, J, g, h)
        evals_small, evecs_small = eigsh(H, 6, which='SA')
        print(evals_small / L)
    elif model == '1dIsing_TE':
        L, J, g, h = sys.argv[2:]
        # Solving -J szsz + g sx + h sz
        L, J, g, h = int(L), float(J), float(g), float(h)
        N = L
        print("python 1dIsing L=%d, J=%f, g=%f, h=%f" % (L, J, g, h))
        H = gen_H_1d_Ising(L, J, g, h)
        H_list = gen_local_H_1d_Ising(L, J, g, h)
        H = np.array(H.todense())

        splus = build_H_one_body([(L//2+1,1)], L, H=None, sx=True, sy=False, sz=False)
        splus = build_H_one_body([(L//2+1,-1j)], L, H=splus, sx=False, sy=True, sz=False)

        dt = 0.05
        exp_iHdt = scipy.linalg.expm(1.j * dt * H)
        total_time = 10

        local_E_array=np.zeros((20, L))

        import matplotlib.pyplot as plt
        for realization in range(10):
            psi = np.exp(np.random.rand(2**L))
            theta = np.random.rand(2**L)*2*np.pi
            psi = psi * np.exp(1j*theta)
            psi = psi/np.linalg.norm(psi)

            psi = splus.dot(psi)
            psi = psi/np.linalg.norm(psi)

            for i in range(int(total_time / 0.05)):
                if i % 10 ==0:
                    print("<E(%.2f)> : " % (i*0.05), psi.conj().T.dot(H).dot(psi))
                    local_E = np.real(measure_local_H(psi, H_list))
                    local_E_array[i//10,:] += local_E
                    print("<local_E(%.2f)> : " % (i*0.05), local_E)
                psi = exp_iHdt.dot(psi)

        for i in range(len(local_E_array)):
            # plt.figure()
            plt.plot(local_E_array[i]/10.)
            # plt.savefig('step_%d.eps' % i)

        plt.legend()
        plt.show()

    else:
        print("error in input arguments:\ncurrently support for 1dJ1J2, 2dAFH")
        raise NotImplementedError


    # sum_Sx = build_Sx(16)
    # x=evecs_small[:,0]
    # print("sum Sx expectation value : ", x.conjugate().dot(sum_Sx.dot(x)))

    for i in range(1,N+1):
        print("Sz at i={0} , {1:.6f} ".format(i, sz_expectation(i, evecs_small[:,0], N)))


    SzSz=[1]
    for i in range(2, N+1):
        SzSz.append(spin_spin_correlation(1, i, N, evecs_small[:,0]))

    SzSz = np.real(np.array(SzSz))
    print("SzSz: ", SzSz)
#     log_file = open('spin_spin_cor_L%d_J2_%d.csv' % (L, J2*10), 'w')
#     np.savetxt(log_file, SzSz/4., '%.4e', delimiter=',')
#     log_file.close()



# plt.plot(np.real(evecs_small[:, 0]), label='real')
# plt.plot(np.imag(evecs_small[:, 0]), label='imag')
# plt.legend()
# plt.show()
