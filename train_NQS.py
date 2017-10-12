from __future__ import absolute_import
from __future__ import print_function

import os
import time
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator
import numpy as np
import tensorflow as tf
from utils.parse_args import parse_args
from network.tf_network import tf_network


"""
1.  Should move config out as an indep class
So that easily to change from 1d problem to 2d problem?
2.  Rewrite the h, J,... etc in a class model
So easily to switch model
"""


class NQS_1d():
    def __init__(self, inputShape, Net, Hamiltonian, batch_size=1):
        self.config = np.zeros((batch_size, inputShape[0], inputShape[1]),
                               dtype=int)
        self.batch_size = batch_size
        self.inputShape = inputShape
        self.init_config(sz0_sector=True)
        self.corrlength = 50

        self.NNet = Net
        self.net_num_para = self.NNet.getNumPara()
        self.moving_E_avg = None

        print("This NQS is aimed for ground state of %s Hamiltonian" % Hamiltonian)
        if Hamiltonian == 'Ising':
            self.get_local_E_batch = self.local_E_Ising_batch
        elif Hamiltonian == 'AFH':
            self.get_local_E_batch = self.local_E_AFH_batch
        elif Hamiltonian == 'J1J2':
            self.get_local_E_batch = self.local_E_J1J2_batch
        else:
            raise NotImplementedError

    def init_config(self, sz0_sector=True):
        if sz0_sector:
            for i in range(self.batch_size):
                x = np.random.randint(2, size=(self.inputShape[0]))
                while(np.sum(x) != self.inputShape[0]/2):
                    x = np.random.randint(2, size=(self.inputShape[0]))

                self.config[i, :, 0] = x
                self.config[i, :, 1] = (x+1) % 2

            return
        else:
            x = np.random.randint(2, size=(self.batch_size, self.inputShape[0]))
            self.config[:, :, 0] = x
            self.config[:, :, 1] = (x+1) % 2
            return

    def getSelfAmp(self):
        return float(self.NNet.forwardPass(self.config))

    def get_self_amp_batch(self):
        return self.NNet.forwardPass(self.config).flatten()

    def eval_amp_array(self, configArray):
        return self.NNet.forwardPass(configArray).flatten()

    def new_config(self):
        L = self.config.shape[1]

        # Restricted to Sz = 0 sectors ##
        randsite1 = np.random.randint(L)
        randsite2 = np.random.randint(L)
        if self.config[0, randsite1, 0] + self.config[0, randsite2, 0] == 1 and randsite1 != randsite2:
            tempconfig = self.config.copy()
            tempconfig[0, randsite1, :] = (tempconfig[0, randsite1, :] + 1) % 2
            tempconfig[0, randsite2, :] = (tempconfig[0, randsite2, :] + 1) % 2
            ratio = self.NNet.forwardPass(tempconfig)[0] / self.getSelfAmp()
            if np.random.rand() < np.amin([1., ratio**2]):
                self.config = tempconfig
            else:
                pass
        else:
            pass


#        tempconfig = self.config.copy()
#        if np.random.rand() < 0.5:
#            randsite = np.random.randint(L)
#            tempconfig[0, randsite, :] = (tempconfig[0, randsite, :] + 1) % 2
#            ratio = self.NNet.forwardPass(tempconfig)[0] / self.getSelfAmp()
#        else:
#            randsite = np.random.randint(L)
#            randsite2 = np.random.randint(L)
#            tempconfig[0, randsite, :] = (tempconfig[0, randsite, :] + 1) % 2
#            tempconfig[0, randsite2, :] = (tempconfig[0, randsite2, :] + 1) % 2
#            ratio = self.NNet.forwardPass(tempconfig)[0] / self.getSelfAmp()
#            if np.random.rand() < np.amin([1., ratio**2]):
#                self.config = tempconfig
#            else:
#                pass

        return

    def new_config_batch(self):
        L = self.config.shape[1]
        batch_size = self.batch_size
        old_amp = self.get_self_amp_batch()

        # Restricted to Sz = 0 sectors ##
        randsite1 = np.random.randint(L, size=(batch_size,))
        randsite2 = np.random.randint(L, size=(batch_size,))
        mask = (self.config[range(batch_size), randsite1, 0] +
                self.config[range(batch_size), randsite2, 0]) == 1

        flip_config = self.config.copy()
        flip_config[range(batch_size), randsite1, :] = (flip_config[range(batch_size), randsite1, :] + 1) % 2
        flip_config[range(batch_size), randsite2, :] = (flip_config[range(batch_size), randsite2, :] + 1) % 2

        ratio = np.power(np.divide(self.eval_amp_array(flip_config), old_amp),  2)
        mask2 = np.random.random_sample((batch_size,)) < ratio
        final_mask = np.logical_and(mask, mask2)
        # update self.config
        # import pdb;pdb.set_trace()
        self.config[final_mask] = flip_config[final_mask]
        return

    def H_exp(self, num_sample=1000, h=1, J=0):
        energyList = []
        correlength = 10
        for i in range(num_sample * correlength):
            self.new_config()
            if i % correlength == 0:
                _, _, localEnergy, _ = self.getLocal(self.config)
                energyList = np.append(energyList, localEnergy)
            else:
                pass
        print(energyList)
        return np.average(energyList)

    def VMC(self, num_sample, iteridx=0, Gj=None, explicit_SR=False):
        L = self.config.shape[1]
        numPara = self.net_num_para
        OOsum = np.zeros((numPara, numPara))
        Osum = np.zeros((numPara))
        Earray = np.zeros((num_sample))
        EOsum = np.zeros((numPara))
        Oarray = np.zeros((numPara, num_sample))

        start_c, start_t = time.clock(), time.time()
        corrlength = self.corrlength
        configDim = list(self.config.shape)
        configDim[0] = num_sample
        configArray = np.zeros(configDim)

        if (self.batch_size == 1):
            for i in range(1, 1 + num_sample * corrlength):
                self.new_config()
                if i % corrlength == 0:
                    configArray[i / corrlength - 1, :, :] = self.config[0, :, :]

        else:
            for i in range(1, 1 + num_sample * corrlength / self.batch_size):
                self.new_config_batch()
                bs = self.batch_size
                if i % corrlength == 0:
                    i_c = i/corrlength
                    configArray[(i_c-1)*bs: i_c*bs, :, :] = self.config[:, :, :]
                else:
                    pass

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time (gen config): ", end_c - start_c, end_t - start_t)

        # for i in range(num_sample):
        #     Earray[i] = self.get_local_E(configArray[i:i+1])

        Earray = self.get_local_E_batch(configArray)

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time ( localE ): ", end_c - start_c, end_t - start_t)

        for i in range(num_sample):
            GList = self.NNet.backProp(configArray[i:i+1])
            Oarray[:, i] = np.concatenate([g.flatten() for g in GList])

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time ( backProp ): ", end_c - start_c, end_t - start_t)

        # Osum = np.einsum('ij->i', Oarray)
        # EOsum = np.einsum('ij,j->i', Oarray, Earray)
        Osum = Oarray.dot(np.ones(Oarray.shape[1]))
        EOsum = Oarray.dot(Earray)

        # for i in range(num_sample):
        #     localO, localE, localEO = self.getLocal_no_OO(configArray[i:i+1])
        #     Osum += localO
        #     Earray[i] = localE
        #     EOsum += localEO
        #     Oarray[:, i] = localO

        if not explicit_SR:
            pass
        else:
            OOsum = Oarray.dot(Oarray.T)

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time (total): ", end_c - start_c, end_t - start_t)
        start_c, start_t = time.clock(), time.time()

        Eavg = np.average(Earray)
        Evar = np.var(Earray)
        # print(self.getSelfAmp())
        print(self.get_self_amp_batch()[:5])
        print("E/N !!!!: ", Eavg / L, "  Var: ", Evar / L / np.sqrt(num_sample))  # , "Earray[:10]",Earray[:10]

        #####################################
        #  Fj = 2<O_iH>-2<H><O_i>
        #####################################
        if self.moving_E_avg is None:
            Fj = 2. * (EOsum / num_sample - Eavg * Osum / num_sample)
        else:
            self.moving_E_avg = self.moving_E_avg * 0.5 + Eavg * 0.5
            Fj = 2. * (EOsum / num_sample - self.moving_E_avg * Osum / num_sample)
            print("moving_E_avg/N !!!!: ", self.moving_E_avg / L)

        if not explicit_SR:
            def implicit_S(v):
                avgO = Osum.flatten()/num_sample
                finalv = - avgO.dot(v) * avgO
                finalv += Oarray.dot((Oarray.T.dot(v)))/num_sample
                return finalv  # + v * 1e-4

            implicit_Sij = LinearOperator((numPara, numPara), matvec=implicit_S)

            Gj, info = scipy.sparse.linalg.minres(implicit_Sij, Fj, x0=Gj)
            print("conv Gj : ", info)
        else:
            #####################################
            # S_ij = <O_i O_j > - <O_i><O_j>   ##
            #####################################
            Sij = OOsum / num_sample - np.einsum('i,j->ij', Osum.flatten(), Osum.flatten()) / (num_sample**2)
            # regu_para = np.amax([10 * (0.9**iteridx), 1e-4])
            # Sij = Sij + regu_para * np.diag(np.ones(Sij.shape[0]))
            Sij = Sij+np.diag(np.ones(Sij.shape[0])*1e-4)
            ############
            # Method 1 #
            ############
            # invSij = np.linalg.inv(Sij)
            # Gj = invSij.dot(Fj.T)
            ############
            # Method 2 #
            ############
            # invSij = np.linalg.pinv(Sij, 1e-3)
            # Gj = invSij.dot(Fj.T)
            ############
            # Method 3 #
            ############
            Gj, info = scipy.sparse.linalg.minres(Sij, Fj, x0=Gj)
            print("conv Gj : ", info)

        # Gj = Fj.T
        print("norm(G): ", np.linalg.norm(Gj),
              "norm(F):", np.linalg.norm(Fj))

        end_c, end_t = time.clock(), time.time()
        print("Sij, Fj time: ", end_c - start_c, end_t - start_t)

        return Gj, Eavg / L, Evar / L / np.sqrt(num_sample)

    def getLocal_no_OO(self, config):
        '''
        forming OO is extremely slow.
        test with np.einsum, np.outer
        '''
        localE = self.get_local_E(config)
        # localE2 = self.local_E_AFH_old(config)
        # if (localE-localE2)>1e-12:
        #     print(np.squeeze(config).T, localE, localE2)

        GList = self.NNet.backProp(config)
        localO = np.concatenate([g.flatten() for g in GList])
        localEO = localO * localE

        return localO, localE, localEO

    # local_E_Ising, get local E from Ising Hamiltonian
    # For only one config.
    def local_E_Ising(self, config, h=1):
        numData, L, inputShape1 = config.shape
        localE = 0.
        for i in range(L - 1):
            temp = config[0, i, :].dot(config[0, i + 1, :])
            localE -= 2 * (temp - 0.5)

        # Periodic Boundary condition
        temp = config[0, 0, :].dot(config[0, -1, :])
        localE -= 2 * (temp - 0.5)
        #####################################

        oldAmp = self.eval_amp_array(config)[0]
        for i in range(L):
            tempConfig = config.copy()
            tempConfig[0, i, :] = (tempConfig[0, i, :] + 1) % 2
            tempAmp = float(self.NNet.forwardPass(tempConfig))
            localE -= h * tempAmp / oldAmp

        return localE

    def local_E_AFH_old(self, config, J=1):
        numData, L, inputShape1 = config.shape
        localE = 0.
        oldAmp = self.eval_amp_array(config)[0]

        for i in range(L - 1):
            # Sz Sz Interaction
            temp = config[0, i, :].dot(config[0, i + 1, :])
            localE += 2 * (temp - 0.5) * J / 4
            if config[0, i, :].dot(config[0, i + 1, :]) == 0:
                tempConfig = config.copy()
                tempConfig[0, i, :] = (tempConfig[0, i, :] + 1) % 2
                tempConfig[0, i + 1, :] = (tempConfig[0, i + 1, :] + 1) % 2
                tempAmp = float(self.NNet.forwardPass(tempConfig))
                localE += J * tempAmp / oldAmp / 2
            else:
                pass
        '''
        Periodic Boundary condition
        '''
        temp = config[0, 0, :].dot(config[0, -1, :])
        localE += 2 * (temp - 0.5) * J / 4
        if temp == 0:
            tempConfig = config.copy()
            tempConfig[0, 0, :] = (tempConfig[0, 0, :] + 1) % 2
            tempConfig[0, L-1, :] = (tempConfig[0, L-1, :] + 1) % 2
            tempAmp = float(self.NNet.forwardPass(tempConfig))
            localE += J * tempAmp / oldAmp / 2

        return localE

    def local_E_AFH(self, config, J=1):
        numData, L, inputShape1 = config.shape
        localE = 0.
        oldAmp = self.eval_amp_array(config)[0]

        # PBC
        config_shift_copy = np.zeros((1, L, inputShape1))
        config_shift_copy[:, :-1, :] = config[:, 1:, :]
        config_shift_copy[:, -1, :] = config[:, 0, :]

        '''
        Sz Sz Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du
        '''
        SzSz = np.einsum('ij,ij->i', config[0, :, :], config_shift_copy[0, :, :])
        localE += np.sum(SzSz - 0.5) * 2 * J / 4

        config_flip = np.einsum('i,ijk->ijk', np.ones(L), config)
        for i in range(L):
            config_flip[i, i, :] = (config_flip[i, i, :] + 1) % 2
            config_flip[i, (i+1) % L, :] = (config_flip[i, (i+1) % L, :] + 1) % 2

#        for i in range(L-1):
#            config_flip[i, i, :] = (config_flip[i, i, :] + 1) % 2
#            config_flip[i, (i+1), :] = (config_flip[i, (i+1), :] + 1) % 2

#        config_flip[L-1, 0, :] = (config_flip[L-1, 0, :] + 1) % 2
#        config_flip[L-1, L-1, :] = (config_flip[L-1, L-1, :] + 1) % 2

        flip_Amp = self.eval_amp_array(config_flip)
        localE += -(SzSz-1).dot(flip_Amp) * J / oldAmp / 2

        return localE

    def local_E_AFH_batch(self, config_arr, J=1):
        '''
        Base on the fact that, in one-hot representation
        Sz Sz Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du
        '''
        num_config, L, inputShape1 = config_arr.shape
        localE_arr = np.zeros((num_config))
        oldAmp = self.eval_amp_array(config_arr)

        # PBC
        config_shift_copy = np.zeros((num_config, L, inputShape1))
        config_shift_copy[:, :-1, :] = config_arr[:, 1:, :]
        config_shift_copy[:, -1, :] = config_arr[:, 0, :]

        # num_config x L
        SzSz = np.einsum('ijk,ijk->ij', config_arr, config_shift_copy)
        localE_arr += np.einsum('ij->i', SzSz - 0.5) * 2 * J / 4

        # num_site(L) x num_config x num_site(L) x num_spin
        config_flip_arr = np.einsum('h,ijk->hijk', np.ones(L), config_arr)
        for i in range(L):
            config_flip_arr[i, :, i, :] = (config_flip_arr[i, :, i, :] + 1) % 2
            config_flip_arr[i, :, (i+1) % L, :] = (config_flip_arr[i, :, (i+1) % L, :] + 1) % 2

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(L*num_config, L, inputShape1))
        flip_Amp_arr = flip_Amp_arr.reshape((L, num_config))
        # localE += -(SzSz-1).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += -np.einsum('ij,ji->i', (SzSz-1), flip_Amp_arr) * J / oldAmp / 2
        return localE_arr

    def local_E_J1J2_batch(self, config_arr, J1=1., J2=1.):
        '''
        Base on the fact that, in one-hot representation
        Sz Sz Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du
        '''
        num_config, L, inputShape1 = config_arr.shape
        localE_arr = np.zeros((num_config))
        oldAmp = self.eval_amp_array(config_arr)

        ####################
        # PBC   J1 term   ##
        ####################
        config_shift_copy = np.zeros((num_config, L, inputShape1))
        config_shift_copy[:, :-1, :] = config_arr[:, 1:, :]
        config_shift_copy[:, -1, :] = config_arr[:, 0, :]

        # num_config x L
        SzSz = np.einsum('ijk,ijk->ij', config_arr, config_shift_copy)
        localE_arr += np.einsum('ij->i', SzSz - 0.5) * 2 * J1 / 4

        # num_site(L) x num_config x num_site(L) x num_spin
        config_flip_arr = np.einsum('h,ijk->hijk', np.ones(L), config_arr)
        for i in range(L):
            config_flip_arr[i, :, i, :] = (config_flip_arr[i, :, i, :] + 1) % 2
            config_flip_arr[i, :, (i+1) % L, :] = (config_flip_arr[i, :, (i+1) % L, :] + 1) % 2

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(L*num_config, L, inputShape1))
        flip_Amp_arr = flip_Amp_arr.reshape((L, num_config))
        # localE += -(SzSz-1).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += -np.einsum('ij,ji->i', (SzSz-1), flip_Amp_arr) * J1 / oldAmp / 2

        ######################
        # PBC  J2 term      ##
        ######################
        config_shift_copy[:, :-2, :] = config_arr[:, 2:, :]
        config_shift_copy[:, -2:, :] = config_arr[:, :2, :]

        # num_config x L
        SzSz = np.einsum('ijk,ijk->ij', config_arr, config_shift_copy)
        localE_arr += np.einsum('ij->i', SzSz - 0.5) * 2 * J2 / 4

        # num_site(L) x num_config x num_site(L) x num_spin
        config_flip_arr = np.einsum('h,ijk->hijk', np.ones(L), config_arr)
        for i in range(L):
            config_flip_arr[i, :, i, :] = (config_flip_arr[i, :, i, :] + 1) % 2
            config_flip_arr[i, :, (i+2) % L, :] = (config_flip_arr[i, :, (i+2) % L, :] + 1) % 2

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(L*num_config, L, inputShape1))
        flip_Amp_arr = flip_Amp_arr.reshape((L, num_config))
        # localE += -(SzSz-1).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += -np.einsum('ij,ji->i', (SzSz-1), flip_Amp_arr) * J2 / oldAmp / 2
        return localE_arr


############################
#  END OF DEFINITION NQS1d #
############################


class NQS_2d():
    def __init__(self, inputShape, Net, Hamiltonian, batch_size=1):
        '''
        config = [batch_size, Lx, Ly, local_dim]
        config represent the product state basis of the model
        in one-hot representation
        Spin-1/2 model: local_dim = 2
        Hubbard model: local_dim = 4
        '''
        self.config = np.zeros((batch_size, inputShape[0], inputShape[1], inputShape[2]),
                               dtype=int)
        self.batch_size = batch_size
        self.inputShape = inputShape
        self.Lx = inputShape[0]
        self.Ly = inputShape[1]
        self.LxLy = self.Lx*self.Ly
        if self.Lx != self.Ly:
            print("not a square lattice !!!")

        self.init_config(sz0_sector=True)
        self.corrlength = 50

        self.NNet = Net
        self.net_num_para = self.NNet.getNumPara()
        self.moving_E_avg = None

        print("This NQS is aimed for ground state of %s Hamiltonian" % Hamiltonian)
        if Hamiltonian == 'Ising':
            raise NotImplementedError
            self.get_local_E_batch = self.local_E_Ising_batch
        elif Hamiltonian == 'AFH':
            self.get_local_E_batch = self.local_E_AFH2d_batch
        elif Hamiltonian == 'J1J2':
            raise NotImplementedError
            self.get_local_E_batch = self.local_E_J1J2_batch
        else:
            raise NotImplementedError

    def init_config(self, sz0_sector=True):
        if sz0_sector:
            for i in range(self.batch_size):
                x = np.random.randint(2, size=(self.Lx, self.Ly))
                while(np.sum(x) != self.LxLy/2):
                    x = np.random.randint(2, size=(self.Lx, self.Ly))

                self.config[i, :, :, 0] = x
                self.config[i, :, :, 1] = (x+1) % 2

            return
        else:
            x = np.random.randint(2, size=(self.batch_size, self.Lx, self.Ly))
            self.config[:, :, :, 0] = x
            self.config[:, :, :, 1] = (x+1) % 2
            return

    def getSelfAmp(self):
        return float(self.NNet.forwardPass(self.config))

    def get_self_amp_batch(self):
        return self.NNet.forwardPass(self.config).flatten()

    def eval_amp_array(self, configArray):
        return self.NNet.forwardPass(configArray).flatten()

    def new_config(self):
        '''
        Implementation for
        1.) random swap transition in spin-1/2 model
        2.) Restricted to Sz = 0 sectors
        '''
        randsite1_x = np.random.randint(self.Lx)
        randsite1_y = np.random.randint(self.Ly)
        randsite2_x = np.random.randint(self.Lx)
        randsite2_y = np.random.randint(self.Ly)
        cond1 = (self.config[0, randsite1_x, randsite1_y, 0] +
                 self.config[0, randsite2_x, randsite2_y, 0]) == 1
        cond2 = randsite1_x != randsite2_x
        cond3 = randsite1_y != randsite2_y
        if cond1 and cond2 and cond3:
            tempconfig = self.config.copy()
            tempconfig[0, randsite1_x, randsite1_y, :] = (tempconfig[0, randsite1_x, randsite1_y, :] + 1) % 2
            tempconfig[0, randsite2_x, randsite2_y, :] = (tempconfig[0, randsite2_x, randsite2_y, :] + 1) % 2
            ratio = self.NNet.forwardPass(tempconfig)[0] / self.getSelfAmp()
            if np.random.rand() < np.amin([1., ratio**2]):
                self.config = tempconfig
            else:
                pass
        else:
            pass

#        tempconfig = self.config.copy()
#        if np.random.rand() < 0.5:
#            randsite = np.random.randint(L)
#            tempconfig[0, randsite, :] = (tempconfig[0, randsite, :] + 1) % 2
#            ratio = self.NNet.forwardPass(tempconfig)[0] / self.getSelfAmp()
#        else:
#            randsite = np.random.randint(L)
#            randsite2 = np.random.randint(L)
#            tempconfig[0, randsite, :] = (tempconfig[0, randsite, :] + 1) % 2
#            tempconfig[0, randsite2, :] = (tempconfig[0, randsite2, :] + 1) % 2
#            ratio = self.NNet.forwardPass(tempconfig)[0] / self.getSelfAmp()
#            if np.random.rand() < np.amin([1., ratio**2]):
#                self.config = tempconfig
#            else:
#                pass

        return

    def new_config_batch(self):
        '''
        Implementation for
        1.) random swap transition in spin-1/2 model
        2.) Restricted to Sz = 0 sectors
        3.) vectorized for batch update
        '''
        batch_size = self.batch_size
        old_amp = self.get_self_amp_batch()

        # Restricted to Sz = 0 sectors ##
        randsite1_x = np.random.randint(self.Lx, size=(batch_size,))
        randsite1_y = np.random.randint(self.Ly, size=(batch_size,))
        randsite2_x = np.random.randint(self.Lx, size=(batch_size,))
        randsite2_y = np.random.randint(self.Ly, size=(batch_size,))
        mask = (self.config[range(batch_size), randsite1_x, randsite1_y, 0] +
                self.config[range(batch_size), randsite2_x, randsite2_y, 0]) == 1

        flip_config = self.config.copy()
        flip_config[range(batch_size), randsite1_x, randsite1_y, :] += 1
        flip_config[range(batch_size), randsite1_x, randsite1_y, :] %= 2
        flip_config[range(batch_size), randsite2_x, randsite2_y, :] += 1
        flip_config[range(batch_size), randsite2_x, randsite2_y, :] %= 2

        ratio = np.power(np.divide(self.eval_amp_array(flip_config), old_amp),  2)
        mask2 = np.random.random_sample((batch_size,)) < ratio
        final_mask = np.logical_and(mask, mask2)
        # update self.config
        self.config[final_mask] = flip_config[final_mask]
        return

    def VMC(self, num_sample, iteridx=0, Gj=None, explicit_SR=False):
        numPara = self.net_num_para
        OOsum = np.zeros((numPara, numPara))
        Osum = np.zeros((numPara))
        Earray = np.zeros((num_sample))
        EOsum = np.zeros((numPara))
        Oarray = np.zeros((numPara, num_sample))

        start_c, start_t = time.clock(), time.time()
        corrlength = self.corrlength
        configDim = list(self.config.shape)
        configDim[0] = num_sample
        configArray = np.zeros(configDim)

        if (self.batch_size == 1):
            for i in range(1, 1 + num_sample * corrlength):
                self.new_config()
                if i % corrlength == 0:
                    configArray[i / corrlength - 1] = self.config[0]

        else:
            for i in range(1, 1 + num_sample * corrlength / self.batch_size):
                self.new_config_batch()
                bs = self.batch_size
                if i % corrlength == 0:
                    i_c = i/corrlength
                    configArray[(i_c-1)*bs: i_c*bs] = self.config[:]
                else:
                    pass

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time (gen config): ", end_c - start_c, end_t - start_t)

        # for i in range(num_sample):
        #     Earray[i] = self.get_local_E(configArray[i:i+1])
        Earray = self.get_local_E_batch(configArray)

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time ( localE ): ", end_c - start_c, end_t - start_t)

        for i in range(num_sample):
            GList = self.NNet.backProp(configArray[i:i+1])
            Oarray[:, i] = np.concatenate([g.flatten() for g in GList])

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time ( backProp ): ", end_c - start_c, end_t - start_t)

        # Osum = np.einsum('ij->i', Oarray)
        # EOsum = np.einsum('ij,j->i', Oarray, Earray)
        Osum = Oarray.dot(np.ones(Oarray.shape[1]))
        EOsum = Oarray.dot(Earray)

        # for i in range(num_sample):
        #     localO, localE, localEO = self.getLocal_no_OO(configArray[i:i+1])
        #     Osum += localO
        #     Earray[i] = localE
        #     EOsum += localEO
        #     Oarray[:, i] = localO

        if not explicit_SR:
            pass
        else:
            OOsum = Oarray.dot(Oarray.T)

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time (total): ", end_c - start_c, end_t - start_t)
        start_c, start_t = time.clock(), time.time()

        Eavg = np.average(Earray)
        Evar = np.var(Earray)
        # print(self.getSelfAmp())
        print(self.get_self_amp_batch()[:5])
        print("E/N !!!!: ", Eavg / self.LxLy, "  Var: ", Evar / self.LxLy / np.sqrt(num_sample))

        #####################################
        #  Fj = 2<O_iH>-2<H><O_i>
        #####################################
        if self.moving_E_avg is None:
            Fj = 2. * (EOsum / num_sample - Eavg * Osum / num_sample)
        else:
            self.moving_E_avg = self.moving_E_avg * 0.5 + Eavg * 0.5
            Fj = 2. * (EOsum / num_sample - self.moving_E_avg * Osum / num_sample)
            print("moving_E_avg/N !!!!: ", self.moving_E_avg / L)

        if not explicit_SR:
            def implicit_S(v):
                avgO = Osum.flatten()/num_sample
                finalv = - avgO.dot(v) * avgO
                finalv += Oarray.dot((Oarray.T.dot(v)))/num_sample
                return finalv  # + v * 1e-4

            implicit_Sij = LinearOperator((numPara, numPara), matvec=implicit_S)

            Gj, info = scipy.sparse.linalg.minres(implicit_Sij, Fj, x0=Gj)
            print("conv Gj : ", info)
        else:
            #####################################
            # S_ij = <O_i O_j > - <O_i><O_j>   ##
            #####################################
            Sij = OOsum / num_sample - np.einsum('i,j->ij', Osum.flatten(), Osum.flatten()) / (num_sample**2)
            # regu_para = np.amax([10 * (0.9**iteridx), 1e-4])
            # Sij = Sij + regu_para * np.diag(np.ones(Sij.shape[0]))
            Sij = Sij+np.diag(np.ones(Sij.shape[0])*1e-4)
            ############
            # Method 1 #
            ############
            # invSij = np.linalg.inv(Sij)
            # Gj = invSij.dot(Fj.T)
            ############
            # Method 2 #
            ############
            # invSij = np.linalg.pinv(Sij, 1e-3)
            # Gj = invSij.dot(Fj.T)
            ############
            # Method 3 #
            ############
            Gj, info = scipy.sparse.linalg.minres(Sij, Fj, x0=Gj)
            print("conv Gj : ", info)

        # Gj = Fj.T
        print("norm(G): ", np.linalg.norm(Gj),
              "norm(F):", np.linalg.norm(Fj))

        end_c, end_t = time.clock(), time.time()
        print("Sij, Fj time: ", end_c - start_c, end_t - start_t)

        return Gj, Eavg / L, Evar / L / np.sqrt(num_sample)

    def local_E_AFH2d_batch(self, config_arr, J=1):
        '''
        Basic idea is due to the fact that
        Sz Sz Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du
        '''
        num_config, Lx, Ly, local_dim = config_arr.shape
        localE_arr = np.zeros((num_config))
        oldAmp = self.eval_amp_array(config_arr)

        # S_ij dot S_(i+1)j
        # PBC
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim))
        config_shift_copy[:, :-1, :, :] = config_arr[:, 1:, :, :]
        config_shift_copy[:, -1, :, :] = config_arr[:, 0, :, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly)), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] += 1
                config_flip_arr[i, j, :, i, j, :] %= 2
                config_flip_arr[i, j, :, (i+1) % L, j, :] += 1
                config_flip_arr[i, j, :, (i+1) % L, j, :] %= 2

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J / oldAmp / 2

        ########################
        # PBC : S_ij dot S_i(j+1)
        ########################
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim))
        config_shift_copy[:, :, :-1, :] = config_arr[:, :, 1:, :]
        config_shift_copy[:, :, -1, :] = config_arr[:, :, 0, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly)), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] += 1
                config_flip_arr[i, j, :, i, j, :] %= 2
                config_flip_arr[i, j, :, i, (j+1) % L, :] += 1
                config_flip_arr[i, j, :, i, (j+1) % L, :] %= 2

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J / oldAmp / 2
        return localE_arr

############################
#  END OF DEFINITION NQS2d #
############################


if __name__ == "__main__":
    ###############################
    #  Read the input argument ####
    ###############################
    alpha_map = {"NN": 2, "NN3": 2, "NN_complex": 1, "NN3_complex": 2,
                 "NN_RBM": 2}

    args = parse_args()
    (L, which_net, lr, num_sample) = (args.L, args.which_net, args.lr, args.num_sample)
    if args.alpha != 0:
        alpha = args.alpha
    else:
        alpha = alpha_map[which_net]

    opt, batch_size, H, dim  = args.opt, args.batch_size, args.H, args.dim
    if dim == 1:
        systemSize = (L, 2)
    elif dim == 2:
        systemSize = (L, L, 2)
    else:
        raise NotImplementedError

    Net = tf_network(which_net, systemSize, optimizer=opt, dim=dim, alpha=alpha)
    if dim == 1:
        N = NQS_1d(systemSize, Net=Net, Hamiltonian=H, batch_size=batch_size)
    else:
        N = NQS_2d(systemSize, Net=Net, Hamiltonian=H, batch_size=batch_size)

    print("Total num para: ", N.net_num_para)
    if N.net_num_para/5 < num_sample:
        print("forming Sij explicitly")
        explicit_SR = True
    else:
        print("DO NOT FORM Sij explicity")
        explicit_SR = False

    var_shape_list = [var.get_shape().as_list() for var in N.NNet.para_list]
    var_list = tf.global_variables()
    saver = tf.train.Saver(N.NNet.model_var_list)

    ckpt_path = 'wavefunction/vmc%dd/%s/L%da%d/' % (dim, which_net, L, alpha)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ckpt = tf.train.get_checkpoint_state(ckpt_path)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(N.NNet.sess, ckpt.model_checkpoint_path)
        print("Restore from last check point")
    else:
        print("No checkpoint found")

    # Thermalization
    print("Thermalizing ~~ ")
    start_t, start_c = time.time(), time.clock()
    if batch_size > 1:
        for i in range(1000):
            N.new_config_batch()
    else:
        for i in range(1000):
            N.new_config()

    end_t, end_c = time.time(), time.clock()
    print("Thermalization time: ", end_c-start_c, end_t-start_t)

    E_log = []
    N.NNet.sess.run(N.NNet.learning_rate.assign(lr))
    N.NNet.sess.run(N.NNet.momentum.assign(0.9))
    GradW, E_avg, E_var = N.VMC(num_sample=num_sample, iteridx=0)
    # N.moving_E_avg = E_avg * l

    for iteridx in range(1, 1000+1):
        print(iteridx)
        '''
        print("Thermalizing ~~ ")
        start_t, start_c = time.time(), time.clock()
        if batch_size > 1:
            for i in range(500):
                N.new_config_batch()
        else:
            for i in range(500):
                N.new_config()

        end_t, end_c = time.time(), time.clock()
        print("Thermalization time: ", end_c-start_c, end_t-start_t)
        '''

        # N.NNet.sess.run(N.NNet.weights['wc1'].assign(wc1))
        # N.NNet.sess.run(N.NNet.biases['bc1'].assign(bc1))

        #    N.NNet.sess.run(N.NNet.learning_rate.assign(1e-3 * (0.995**iteridx)))
        #    N.NNet.sess.run(N.NNet.momentum.assign(0.95 - 0.4 * (0.98**iteridx)))
        # num_sample = 500 + iteridx/10

        GradW, E, E_var = N.VMC(num_sample=num_sample, iteridx=iteridx,
                                Gj=GradW, explicit_SR=explicit_SR)
        # GradW = GradW/np.linalg.norm(GradW)*np.amax([(0.95**iteridx),0.1])
        if np.linalg.norm(GradW) > 1000:
            GradW = GradW/np.linalg.norm(GradW)

        E_log.append(E)
        grad_list = []
        grad_ind = 0
        for var_shape in var_shape_list:
            var_size = np.prod(var_shape)
            grad_list.append(GradW[grad_ind:grad_ind + var_size].reshape(var_shape))
            grad_ind += var_size

        #  L2 Regularization ###
        # for idx, W in enumerate(N.NNet.sess.run(N.NNet.para_list)):
        #     grad_list[idx] += W * 0.001

        N.NNet.applyGrad(grad_list)
        # To save object ##
        if iteridx % 50 == 0:
            saver.save(N.NNet.sess, ckpt_path + 'opt%s_S%d' % (opt, num_sample))

    log_file = open('L%d_%s_a%s_%s%.e_S%d.csv' % (L, which_net, alpha, opt, lr, num_sample),
                    'a')
    np.savetxt(log_file,
               E_log, '%.4e', delimiter=',')
    log_file.close()
    '''
    Task1
    Write down again the Probability assumption
    and the connection with deep learning model

    '''
