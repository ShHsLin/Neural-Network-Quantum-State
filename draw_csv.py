import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import setup

## [TODO] add if "AFH" in filename, to distinguish different model
## [TODO] separate this outside as data file
E0_dict = {4: - 0.57432544,
           6: - 0.6035218,
           8: - 0.619033,
           10: -0.628656,
          }

E1_dict = {4: -0.54293359, # (From ED ) (not sure about its \sum_i<Sz_i> = 0 ?
           6: -0.5942205111111111,  # ( from the gap 1/L estimate)
           8: -0.61510901875, #(From peps not exact)
           10: -0.6266469216,
          }


conv_int = 25

filename = sys.argv[1]
# E_array=np.genfromtxt(filename, dtype=str)
# E_array=np.loadtxt(filename).view(complex)

fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.5), sharex=True)
plt.subplots_adjust(hspace=0.08)

for filename in sys.argv[1:]:
    if filename[-4:] == '.npy':
        E_array = np.load(filename)
    elif filename[-4:] == '.pkl':
        result = pickle.load(open(filename, 'rb'))
        E_array = result['E0']
        Evar_array = result['E0_var']
        max_amp_array = np.abs(result['max_amp'])

    else:
        E_array=np.loadtxt(filename, dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})


    print(np.sum(E_array))
    # plt.plot(np.real(E_array))
    # plt.plot(np.convolve(np.real(E_array), [1./conv_int]*conv_int, 'valid'))
    try:
        idx = filename.find('L')
        L = int(filename[idx+1:idx+3])
    except:
        idx = filename.find('L')
        L = int(filename[idx+1:idx+2])

    if 'splus' in filename:
        E0 = E1_dict[L]
    else:
        E0 = E0_dict[L]

    axes[0].semilogy(np.convolve(np.real(E_array)-E0, [1./conv_int]*conv_int, 'valid'))
    axes[0].set_ylabel('E/N')
    # plt.plot(np.convolve(np.real(E_array)-E0, [1./25]*25, 'valid'))
    axes[1].semilogy(np.convolve(Evar_array, [1./conv_int]*conv_int, 'valid'))
    axes[1].set_ylabel('Var(E/N)')
    axes[2].plot(np.convolve(max_amp_array, [1./conv_int]*conv_int, 'valid'))

plt.subplots_adjust(left=0.25, bottom=0.1, top=0.93, right=0.98)
plt.show()


