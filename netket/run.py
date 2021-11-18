# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import netket as nk
from ed import load_ed_data
from ed import load_ed_data2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os, sys, time

L = int(sys.argv[1])
T = float(sys.argv[2])
alpha = int(sys.argv[3])
method = 'sr'
str_lr = '1e-1'
seed = int(sys.argv[4])

np.random.seed(seed)

# Load the Hilbert space info and data
hi, training_samples, training_targets = load_ed_data2(L, T)
# hi, training_samples, training_targets = load_ed_data(L)


# Machine
ma = nk.machine.RbmSpin(hilbert=hi, alpha=alpha)
ma.init_random_parameters(seed=seed, sigma=0.01)


# Optimizer
overlaps = []
timesteps = []
# for lr in [1e-3, 1e-4]:
# for op in [nk.optimizer.AdaDelta()]:
# nk.optimizer.Momentum(learning_rate=1e-4)]:
if not os.path.exists('results/'):
    os.makedirs('results/')


# op = nk.optimizer.AdaDelta()

if method == 'Gd':
    for lr in [1e-3, 1e-4]:
        op = nk.optimizer.RmsProp(learning_rate=lr)
        # op = nk.optimizer.Sgd(learning_rate=1e-2)

        # Stochastic Reconfiguration
        # sr = nk.optimizer.SR(diag_shift=0.1, use_iterative=False,
        #                      lsq_solver="LLT")

        method = 'Gd'

        spvsd = nk.supervised.Supervised(
            machine=ma,
            optimizer=op,
            method=method,
            batch_size=1000,
            samples=training_samples,
            targets=training_targets,
        )

        n_iter = 2000


        # Run with "Overlap_phi" loss. Also available currently is "MSE", "Overlap_uni"
        for i in range(1, 1+n_iter):
            spvsd.advance(loss_function="Overlap_phi")
            overlaps.append(np.exp(-spvsd.loss_log_overlap))
            timesteps.append(time.time() - t0)
            if i % 100 == 0:
                print(" idx = ", i)
                np_overlaps = np.array(overlaps)
                np_timesteps = np.array(timesteps)
                np.save('overlaps_L%d_T%.2f_alpha%d_%s_RmsProp.npy' % (L, T, alpha, method), overlaps)
else:
    assert method == 'sr'
    lr = float(str_lr)
    op = nk.optimizer.Sgd(learning_rate=lr)

    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=0.1, use_iterative=True,
                         lsq_solver="LLT")

    spvsd = nk.supervised.Supervised(
        machine=ma,
        optimizer=op,
        method=method,
        batch_size=1000,
        samples=training_samples,
        targets=training_targets,
    )

    n_iter = 2000


    t0 = time.time()
    # Run with "Overlap_phi" loss. Also available currently is "MSE", "Overlap_uni"
    for i in range(1, 1+n_iter):
        spvsd.advance(loss_function="Overlap_phi")
        overlaps.append(np.exp(-spvsd.loss_log_overlap))
        timesteps.append(time.time() - t0)
        if i % 50 == 0:
            print(" idx = ", i, overlaps[-1], timesteps[-1])
            np_overlaps = np.array(overlaps)
            np_timesteps = np.array(timesteps)
            data_dict = {'overlaps': np_overlaps, 'timesteps': np_timesteps}
            # np.save('overlaps_L%d_T%.2f_alpha%d_%s.npy' % (L, T, alpha, method), overlaps)
            pickle.dump(data_dict, open('results/overlaps_L%d_T%.2f_alpha%d_%s_lr%s_seed%d.pkl' % (L, T, alpha, method, str_lr, seed), 'wb'))




if nk.MPI.rank() == 0:
    plt.semilogy(1. - np_overlaps)
    plt.ylabel("Overlap")
    plt.xlabel("Iteration #")
    plt.axhline(y=1e-4, xmin=0, xmax=n_iter, linewidth=2, color="k", label="1")
    plt.title(r"Transverse-field Ising model, $L=" + str(L) + "$")
    plt.show()
