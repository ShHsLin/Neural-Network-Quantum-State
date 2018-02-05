import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator
import numpy as np
import tensorflow as tf
import time


"""
Should add spin-spin correlation in 2d
should add vmc_observable in 2d

1.  Should move config out as an indep class
So that easily to change from 1d problem to 2d problem?
2.  Rewrite the h, J,... etc in a class model
So easily to switch model
"""


class NQS_1d():
    def __init__(self, inputShape, Net, Hamiltonian, batch_size=1, J2=None, reg=0.):
        self.config = np.zeros((batch_size, inputShape[0], inputShape[1]),
                               dtype=int)
        self.batch_size = batch_size
        self.inputShape = inputShape
        self.init_config(sz0_sector=True)
        self.corrlength = 50
        self.max_batch_size = 5000
        if self.batch_size > self.max_batch_size:
            print("batch_size > max_batch_size, memory error may occur")

        self.NNet = Net
        self.net_num_para = self.NNet.getNumPara()
        self.moving_E_avg = None
        self.reg = reg

        print("This NQS is aimed for ground state of %s Hamiltonian" % Hamiltonian)
        if Hamiltonian == 'Ising':
            self.get_local_E_batch = self.local_E_Ising_batch
        elif Hamiltonian == 'AFH':
            self.get_local_E_batch = self.local_E_AFH_batch
        elif Hamiltonian == 'J1J2':
            self.J2 = J2
            self.get_local_E_batch = self.local_E_J1J2_batch
        else:
            raise NotImplementedError

    def init_config(self, sz0_sector=True):
        if sz0_sector:
            for i in range(self.batch_size):
                x = np.random.randint(2, size=(self.inputShape[0]))
                while(np.sum(x) != self.inputShape[0]//2):
                    x = np.random.randint(2, size=(self.inputShape[0]))

                self.config[i, :, 0] = x
                self.config[i, :, 1] = (1 - x)

            return
        else:
            x = np.random.randint(2, size=(self.batch_size, self.inputShape[0]))
            self.config[:, :, 0] = x
            self.config[:, :, 1] = (1 - x)
            return

    def getSelfAmp(self):
        return float(self.NNet.forwardPass(self.config))

    def get_self_amp_batch(self):
        return self.NNet.forwardPass(self.config).flatten()

    def eval_amp_array(self, configArray):
        # (batch_size, inputShape[0], inputShape[1])
        array_shape = configArray.shape
        max_size = self.max_batch_size
        if array_shape[0] <= max_size:
            return self.NNet.forwardPass(configArray).flatten()
        else:
            amp_array = np.empty((array_shape[0], ), dtype=np.float64)
            for idx in range(array_shape[0] // max_size):
                amp_array[max_size * idx : max_size * (idx + 1)] = self.NNet.forwardPass(configArray[max_size * idx : max_size * (idx + 1)]).flatten()

            amp_array[max_size * (array_shape[0]//max_size) : ] = self.NNet.forwardPass(configArray[max_size * (array_shape[0]//max_size) : ]).flatten()
            return amp_array

    def new_config(self):
        L = self.config.shape[1]

        # Restricted to Sz = 0 sectors ##
        randsite1 = np.random.randint(L)
        randsite2 = np.random.randint(L)
        if self.config[0, randsite1, 0] + self.config[0, randsite2, 0] == 1 and randsite1 != randsite2:
            tempconfig = self.config.copy()
            tempconfig[0, randsite1, :] = (1 - tempconfig[0, randsite1, :])
            tempconfig[0, randsite2, :] = (1 - tempconfig[0, randsite2, :])
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
        mask = (self.config[np.arange(batch_size), randsite1, 0] +
                self.config[np.arange(batch_size), randsite2, 0]) == 1

        flip_config = self.config.copy()
        flip_config[np.arange(batch_size), randsite1, :] = (1 - flip_config[np.arange(batch_size), randsite1, :])
        flip_config[np.arange(batch_size), randsite2, :] = (1 - flip_config[np.arange(batch_size), randsite2, :])

        ratio = np.divide(np.abs(self.eval_amp_array(flip_config))+1e-45, np.abs(old_amp)+1e-45 )
        ratio_square = np.power(ratio,  2)
        mask2 = np.random.random_sample((batch_size,)) < ratio_square

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

    def VMC_observable(self, num_sample):
        L = self.config.shape[1]
        numPara = self.net_num_para

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

        SzSz = self.spin_spin_correlation(configArray)

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time ( spin-spin-correlation ): ", end_c - start_c, end_t - start_t)
        return SzSz

    def VMC(self, num_sample, iteridx=0, SR=True, Gj=None, explicit_SR=False):
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
            for i in range(1, 1 + int(num_sample * corrlength / self.batch_size)):
                self.new_config_batch()
                if i % corrlength == 0:
                    bs = self.batch_size
                    i_c = int(i/corrlength)
                    configArray[(i_c-1)*bs: i_c*bs, :, :] = self.config[:, :, :]
                else:
                    pass

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time (gen config): ", end_c - start_c, end_t - start_t)

        # for i in range(num_sample):
        #     Earray[i] = self.get_local_E(configArray[i:i+1])

        Earray = self.get_local_E_batch(configArray)
        ####################################
        ## TO perform adiabatic H0 --> H1 ##
        ####################################
        # Earray = self.local_E_J1J2_batch(configArray, J1=1., J2=0.5 + iteridx*0.0005)

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time ( localE ): ", end_c - start_c, end_t - start_t)
        if not SR:
            Eavg = np.average(Earray)
            Evar = np.var(Earray)
            print(self.get_self_amp_batch()[:5])
            print("E/N !!!!: ", Eavg / L, "  Var: ", Evar / L / np.sqrt(num_sample))  # , "Earray[:10]",Earray[:10]
            Glist = self.NNet.vanilla_back_prop(configArray, Earray)
            # Reg
            for idx, W in enumerate(self.NNet.sess.run(self.NNet.para_list)):
                Glist[idx] += W * self.reg

            Gj = np.concatenate([g.flatten() for g in Glist]) * 2./num_sample
            end_c, end_t = time.clock(), time.time()
            print("monte carlo time ( backProp ): ", end_c - start_c, end_t - start_t)
            print("norm(G): ", np.linalg.norm(Gj))
            return Gj, Eavg / L, Evar / L / np.sqrt(num_sample)
        else:
            pass

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
        print(self.get_self_amp_batch()[:5])
        print("E/N !!!!: ", Eavg / L, "  Var: ", Evar / L / np.sqrt(num_sample))

        #####################################
        #  Fj = 2<O_iH>-2<H><O_i>
        #####################################
        if self.moving_E_avg is None:
            Fj = 2. * (EOsum / num_sample - Eavg * Osum / num_sample)
            Fj += self.reg * np.concatenate([g.flatten() for g in self.NNet.sess.run(self.NNet.para_list)])
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
            tempConfig[0, i, :] = (1 - tempConfig[0, i, :])
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
                tempConfig[0, i, :] = (1 - tempConfig[0, i, :])
                tempConfig[0, i + 1, :] = (1 - tempConfig[0, i + 1, :])
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
            tempConfig[0, 0, :] = (1 - tempConfig[0, 0, :])
            tempConfig[0, L-1, :] = (1 - tempConfig[0, L-1, :])
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
            config_flip[i, i, :] = (1 - config_flip[i, i, :])
            config_flip[i, (i+1) % L, :] = (1 - config_flip[i, (i+1) % L, :])

#        for i in range(L-1):
#            config_flip[i, i, :] = (config_flip[i, i, :] + 1) % 2
#            config_flip[i, (i+1), :] = (config_flip[i, (i+1), :] + 1) % 2

#        config_flip[L-1, 0, :] = (config_flip[L-1, 0, :] + 1) % 2
#        config_flip[L-1, L-1, :] = (config_flip[L-1, L-1, :] + 1) % 2

        flip_Amp = self.eval_amp_array(config_flip)
        localE += -(SzSz-1).dot(flip_Amp) * J / oldAmp / 2

        return localE

    def spin_spin_correlation(self, config_arr):
        '''
        Compute the spin-spin correlation <S_i S_j>
        without average over i
        '''
        num_config, L, inputShape1 = config_arr.shape

        config_shift_copy = np.zeros((num_config, L, inputShape1))
        SzSz_j = [0.25]
        for j in range(1,L):
            config_shift_copy[:, :-j, :] = config_arr[:, j:, :]
            config_shift_copy[:, -j:, :] = config_arr[:, :j, :]
            SzSz = np.einsum('ijk,ijk->', config_arr, config_shift_copy) / L / num_config
            # SzSz average over L and num_config
            # In this convention SzSz = 1 or 0
            SzSz_j.append((SzSz-0.5)/2.)

        return(np.array(SzSz_j))

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
            config_flip_arr[i, :, i, :] = (1 - config_flip_arr[i, :, i, :])
            config_flip_arr[i, :, (i+1) % L, :] = (1 - config_flip_arr[i, :, (i+1) % L, :])

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(L*num_config, L, inputShape1))
        flip_Amp_arr = flip_Amp_arr.reshape((L, num_config))
        # localE += -(SzSz-1).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += -np.einsum('ij,ji->i', (SzSz-1), flip_Amp_arr) * J / oldAmp / 2
        return localE_arr

    def local_E_J1J2_batch(self, config_arr):
        J1 = 1.
        J2 = self.J2
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
            config_flip_arr[i, :, i, :] = (1 - config_flip_arr[i, :, i, :])
            config_flip_arr[i, :, (i+1) % L, :] = (1 - config_flip_arr[i, :, (i+1) % L, :])

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
            config_flip_arr[i, :, i, :] = (1 - config_flip_arr[i, :, i, :])
            config_flip_arr[i, :, (i+2) % L, :] = (1 - config_flip_arr[i, :, (i+2) % L, :])

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(L*num_config, L, inputShape1))
        flip_Amp_arr = flip_Amp_arr.reshape((L, num_config))
        # localE += -(SzSz-1).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += -np.einsum('ij,ji->i', (SzSz-1), flip_Amp_arr) * J2 / oldAmp / 2
        return localE_arr


############################
#  END OF DEFINITION NQS1d #
############################


class NQS_2d():
    def __init__(self, inputShape, Net, Hamiltonian, batch_size=1, J2=None, reg=0.):
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
        self.max_batch_size = 5000
        if self.batch_size > self.max_batch_size:
            print("batch_size > max_batch_size, memory error may occur")

        self.NNet = Net
        self.net_num_para = self.NNet.getNumPara()
        self.moving_E_avg = None
        self.reg = reg

        print("This NQS is aimed for ground state of %s Hamiltonian" % Hamiltonian)
        if Hamiltonian == 'Ising':
            raise NotImplementedError
            self.get_local_E_batch = self.local_E_Ising_batch
        elif Hamiltonian == 'AFH':
            self.get_local_E_batch = self.local_E_2dAFH_batch
        elif Hamiltonian == 'J1J2':
            self.J2 = J2
            self.get_local_E_batch = self.local_E_2dJ1J2_batch
        else:
            raise NotImplementedError

    def init_config(self, sz0_sector=True):
        if sz0_sector:
            for i in range(self.batch_size):
                x = np.random.randint(2, size=(self.Lx, self.Ly))
                while(np.sum(x) != self.LxLy//2):
                    x = np.random.randint(2, size=(self.Lx, self.Ly))

                self.config[i, :, :, 0] = x
                self.config[i, :, :, 1] = (1 - x)

            return
        else:
            x = np.random.randint(2, size=(self.batch_size, self.Lx, self.Ly))
            self.config[:, :, :, 0] = x
            self.config[:, :, :, 1] = (1 - x)
            return

    def getSelfAmp(self):
        return float(self.NNet.forwardPass(self.config))

    def get_self_amp_batch(self):
        return self.NNet.forwardPass(self.config).flatten()

    def eval_amp_array(self, configArray):
        # (batch_size, inputShape[0], inputShape[1], inputShape[2])
        array_shape = configArray.shape
        max_size = self.max_batch_size
        if array_shape[0] <= max_size:
            # import pdb;pdb.set_trace()
            return self.NNet.forwardPass(configArray).flatten()
        else:
            amp_array = np.empty((array_shape[0], ), dtype=np.float64)
            for idx in range(array_shape[0] // max_size):
                amp_array[max_size * idx : max_size * (idx + 1)] = self.NNet.forwardPass(configArray[max_size * idx : max_size * (idx + 1)]).flatten()

            amp_array[max_size * (array_shape[0]//max_size) : ] = self.NNet.forwardPass(configArray[max_size * (array_shape[0]//max_size) : ]).flatten()
            # import pdb;pdb.set_trace()
            return amp_array

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
            tempconfig[0, randsite1_x, randsite1_y, :] = (1 - tempconfig[0, randsite1_x, randsite1_y, :])
            tempconfig[0, randsite2_x, randsite2_y, :] = (1 - tempconfig[0, randsite2_x, randsite2_y, :])
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
        4.) 10% propasal include global spin inversion
        '''
        batch_size = self.batch_size
        old_amp = self.get_self_amp_batch()

        # Restricted to Sz = 0 sectors ##
        randsite1_x = np.random.randint(self.Lx, size=(batch_size,))
        randsite1_y = np.random.randint(self.Ly, size=(batch_size,))
        # Random Update
        # randsite2_x = np.random.randint(self.Lx, size=(batch_size,))
        # randsite2_y = np.random.randint(self.Ly, size=(batch_size,))
        # Local Update, (0) right (1) upper right (2) up 
        rand_direct = np.random.randint(3, size=(batch_size,))
        rand_dx = 1 - (rand_direct // 2)
        rand_dy = (rand_direct + 1) // 2
        randsite2_x = np.array( (randsite1_x + self.Lx + rand_dx) % self.Lx, dtype=np.int32)
        randsite2_y = np.array( (randsite1_y + self.Ly + rand_dy) % self.Ly, dtype=np.int32)

        mask = (self.config[np.arange(batch_size), randsite1_x, randsite1_y, 0] +
                self.config[np.arange(batch_size), randsite2_x, randsite2_y, 0]) == 1

        flip_config = self.config.copy()
        flip_config[np.arange(batch_size), randsite1_x, randsite1_y, :] = 1 - flip_config[np.arange(batch_size), randsite1_x, randsite1_y, :]
        flip_config[np.arange(batch_size), randsite2_x, randsite2_y, :] = 1 - flip_config[np.arange(batch_size), randsite2_x, randsite2_y, :]

        # Random total spin flip 10 % of configurations
        to_flip_idx = np.random.choice(batch_size, batch_size//10, replace=False)
        flip_config[to_flip_idx] = 1 - flip_config[to_flip_idx]

        ratio = np.zeros((batch_size,))
        ratio[mask] = np.divide(np.abs(self.eval_amp_array(flip_config[mask]))+1e-45, np.abs(old_amp[mask])+1e-45 )
        ratio_square = np.power(ratio,  2)
        mask2 = np.random.random_sample((batch_size,)) < ratio_square
        final_mask = np.logical_and(mask, mask2)
        # update self.config
        self.config[final_mask] = flip_config[final_mask]

        acceptance_ratio = np.sum(final_mask)/batch_size
        return acceptance_ratio

    def VMC(self, num_sample, iteridx=0, SR=True, Gj=None, explicit_SR=False):
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
            sum_accept_ratio = 0
            for i in range(1, 1 + int(num_sample * corrlength / self.batch_size)):
                ac = self.new_config_batch()
                sum_accept_ratio += ac
                bs = self.batch_size
                if i % corrlength == 0:
                    i_c = int(i/corrlength)
                    configArray[(i_c-1)*bs: i_c*bs] = self.config[:]
                else:
                    pass

            print("acceptance ratio: ", sum_accept_ratio/(1. + int(num_sample * corrlength / self.batch_size)))

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time (gen config): ", end_c - start_c, end_t - start_t)

        # for i in range(num_sample):
        #     Earray[i] = self.get_local_E(configArray[i:i+1])
        Earray = self.get_local_E_batch(configArray)

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time ( localE ): ", end_c - start_c, end_t - start_t)

        if not SR:
            Eavg = np.average(Earray)
            Evar = np.var(Earray)
            print(self.get_self_amp_batch()[:5])
            print("E/N !!!!: ", Eavg / self.LxLy, "  Var: ", Evar / self.LxLy / np.sqrt(num_sample))
            Glist = self.NNet.vanilla_back_prop(configArray, Earray)
            # Reg
            for idx, W in enumerate(self.NNet.sess.run(self.NNet.para_list)):
                Glist[idx] += W * self.reg

            Gj = np.concatenate([g.flatten() for g in Glist]) * 2./num_sample
            end_c, end_t = time.clock(), time.time()
            print("monte carlo time ( backProp ): ", end_c - start_c, end_t - start_t)
            print("norm(G): ", np.linalg.norm(Gj))
            return Gj, Eavg / self.LxLy, Evar / self.LxLy / np.sqrt(num_sample)
        else:
            pass

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
            Fj += self.reg * np.concatenate([g.flatten() for g in self.NNet.sess.run(self.NNet.para_list)])

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

        return Gj, Eavg / self.LxLy, Evar / self.LxLy / np.sqrt(num_sample)

    def local_E_2dAFH_batch(self, config_arr, J=1):
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
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, j, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, j, :]

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
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, i, (j+1) % Ly, :] = 1 - config_flip_arr[i, j, :, i, (j+1) % Ly, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J / oldAmp / 2
        return localE_arr

    def local_E_2dJ1J2_batch(self, config_arr):
        J1 = 1.
        J2 = self.J2
        '''
        Basic idea is due to the fact that
        Sz Sz Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du
        '''
        num_config, Lx, Ly, local_dim = config_arr.shape
        localE_arr = np.zeros((num_config))
        oldAmp = self.eval_amp_array(config_arr)


        ########################
        # PBC : J1 S_ij dot S_(i+1)j
        ########################
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim))
        config_shift_copy[:, :-1, :, :] = config_arr[:, 1:, :, :]
        config_shift_copy[:, -1, :, :] = config_arr[:, 0, :, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J1 / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly)), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, j, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, j, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J1 / oldAmp / 2
        localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J1 / oldAmp / 2

        ########################
        # PBC : J1 S_ij dot S_i(j+1)
        ########################
        # config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim))
        config_shift_copy[:, :, :-1, :] = config_arr[:, :, 1:, :]
        config_shift_copy[:, :, -1, :] = config_arr[:, :, 0, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J1 / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly)), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, i, (j+1) % Ly, :] = 1 - config_flip_arr[i, j, :, i, (j+1) % Ly, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J1 / oldAmp / 2

        ########################
        # PBC : J2 S_ij dot S_(i+1)(j+1)
        ########################
        # moving the origin config 1 down and 1 left
        config_shift_copy[:, :-1, :-1, :] = config_arr[:, 1:, 1:, :]
        config_shift_copy[:, :-1, -1, :] = config_arr[:, 1:, 0, :]
        config_shift_copy[:, -1, :-1, :] = config_arr[:, 0, 1:, :]
        config_shift_copy[:, -1, -1, :] = config_arr[:, 0, 0, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J2 / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly)), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, (j+1) % Ly, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, (j+1) % Ly, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J2 / oldAmp / 2

        ########################
        # PBC : J2 S_ij dot S_(i+1)(j-1)
        ########################
        # moving the origin config 1 up and 1 left
        config_shift_copy[:, :-1, 1:, :] = config_arr[:, 1:, :-1, :]
        config_shift_copy[:, -1, 1:, :] = config_arr[:, 0, :-1, :]
        config_shift_copy[:, :-1, 0, :] = config_arr[:, 1:, -1, :]
        config_shift_copy[:, -1, 0, :] = config_arr[:, 0, -1, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J2 / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly)), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, (j-1+Ly) % Ly, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, (j-1+Ly) % Ly, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J2 / oldAmp / 2

        return localE_arr



############################
#  END OF DEFINITION NQS2d #
############################
