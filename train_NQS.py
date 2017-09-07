from __future__ import absolute_import
from __future__ import print_function

import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator
import numpy as np
import tensorflow as tf
import pickle

from utils.parse_args import parse_args
from utils.prepare_net import prepare_net

import time

"""
1.  Should move config out as an indep class
So that easily to change from 1d problem to 2d problem?
2.  Rewrite the h, J,... etc in a class model
So easily to switch model
"""


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def read_object(filename):
    with open(filename, 'r') as input:
        return pickle.load(input)


class NQS():
    def __init__(self, inputShape, Net, Hamiltonian, batch_size=1):
        self.config = np.zeros((batch_size, inputShape[0], inputShape[1]),
                               dtype=int)
        self.batch_size = batch_size
        self.inputShape = inputShape
        self.init_config(sz0_sector=True)

        self.NNet = Net
        self.moving_E_avg = None

        self.ampDic = {}
        if Hamiltonian == 'Ising':
            self.get_local_E = self.local_E_Ising
        elif Hamiltonian == 'AFH':
            self.get_local_E = self.local_E_AFH

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

    def getNumPara(self):
        return self.NNet.getNumPara()

    def getSelfAmp(self):
        return float(self.NNet.forwardPass(self.config))
        # configStr = ''.join([str(ele) for ele in self.config.flatten()])
        # if configStr in self.ampDic:
        #     return self.ampDic[configStr]
        # else:
        #     amp = float(self.NNet.forwardPass(self.config))
        #     self.ampDic[configStr] = amp
        #     return amp

    def get_self_amp_batch(self):
        return self.NNet.forwardPass(self.config).flatten()

    def eval_amp_array(self, configArray):
        return self.NNet.forwardPass(configArray).flatten()
        # numData, inputShape0, inputShape1 = configArray.shape
        # ampArray = np.zeros((numData))
        # for i in range(numData):
        #     config = configArray[i, :, :]
        #     configStr = ''.join([str(ele) for ele in config.flatten()])
        #     if configStr in self.ampDic:
        #         amp = self.ampDic[configStr]
        #     else:
        #         amp = float(self.NNet.forwardPass(config))
        #         self.ampDic[configStr] = amp

        #     ampArray[i] = amp

        # return ampArray

    def cleanAmpDic(self):
        self.ampDic = {}

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
        self.cleanAmpDic()

        L = self.config.shape[1]
        numPara = self.getNumPara()
        OOsum = np.zeros((numPara, numPara))
        Osum = np.zeros((numPara))
        Earray = np.zeros((num_sample))
        EOsum = np.zeros((numPara))
        Oarray = np.zeros((numPara, num_sample))

        start_c, start_t = time.clock(), time.time()
        if self.batch_size > 100:
            corrlength = 50
        else:
            corrlength = 15
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

        for i in range(num_sample):
            # localO, localOO, localE, localEO = self.getLocal(configArray[i:i+1])
            localO, localE, localEO = self.getLocal_no_OO(configArray[i:i+1])
            Osum += localO
            Earray[i] = localE
            EOsum += localEO
            # OOsum += localOO
            Oarray[:, i] = localO

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
            ############
            # Method 2 #
            ############
            # invSij = np.linalg.pinv(Sij, 1e-6)
            ############
            # Method 3 #
            ############
            Gj, info = scipy.sparse.linalg.minres(Sij, Fj, x0=Gj)
            print("conv Gj : ", info)

        # Gj = invSij.dot(Fj.T)
        # Gj = Fj.T
        print("norm(G): ", np.linalg.norm(Gj),
              "norm(F):", np.linalg.norm(Fj))

        end_c, end_t = time.clock(), time.time()
        print("Sij, Fj time: ", end_c - start_c, end_t - start_t)

        return Gj, Eavg / L, Evar / L / np.sqrt(num_sample)

    def getLocal(self, config):
        localE = self.get_local_E(config)

        GList = self.NNet.backProp(config)
        localO = np.concatenate([g.flatten() for g in GList])

        localOO = np.einsum('i,j->ij', localO, localO)
        localEO = localO * localE

        return localO, localOO, localE, localEO

    def getLocal_no_OO(self, config):
        '''
        forming OO is extremely slow.
        test with np.einsum, np.outer
        '''
        # localE = self.get_local_E(config)
        localE = self.local_E_AFH_new(config)
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

    def local_E_AFH(self, config, J=1):
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

    def local_E_AFH_new(self, config, J=1):
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


########################
#  END OF DEFINITION  #
########################


if __name__ == "__main__":

    alpha_map = {"NN": 2, "NN3": 2, "NN_complex": 1, "NN3_complex": 2,
                 "NN_RBM": 2}

    args = parse_args()
    L = args.L
    which_net = args.which_net
    lr = args.lr
    num_sample = args.num_sample
    if args.alpha != 0:
        alpha = args.alpha
    else:
        alpha = alpha_map[which_net]

    opt = args.opt
    batch_size = args.batch_size
    H = 'AFH'
    systemSize = (L, 2)

    Net = prepare_net(which_net, systemSize, opt, alpha)
    net_num_para = Net.getNumPara()
    print("Total num para: ", net_num_para)
    if net_num_para < num_sample:
        explicit_SR = False
    else:
        explicit_SR = True

    N = NQS(systemSize, Net=Net, Hamiltonian=H, batch_size=batch_size)

    var_shape_list = [var.get_shape().as_list() for var in N.NNet.para_list]
    var_list = tf.global_variables()
    saver = tf.train.Saver(N.NNet.model_var_list)
    ckpt = tf.train.get_checkpoint_state('Model/VMC/'+which_net+'/L'+str(L)+'/')

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

        N.NNet.applyGrad(grad_list)
        #  L2 Regularization ###
        # for idx, W in enumerate( N.NNet.sess.run(N.NNet.para_list)):
        #        GList[idx] += W*0.1

        # To save object ##
        if iteridx % 50 == 0:
            saver.save(N.NNet.sess, 'Model/VMC/'+which_net+'/L'+str(L)+'/pre')

    # np.savetxt('Ising_CNN2_Mom/%.e.csv' % N.NNet.learning_rate.eval(N.NNet.sess),
    #           E_log, '%.4e', delimiter=',')

    # save_object(N,'NNQS_AFH_L40_Mom.obj')
    # save_object(N,'CNNQS_AFH_L40_noMom.obj')

    '''
    Task0
    Rewrite it as batch to improve the speed

    Task1
    Write down again the Probability assumption
    and the connection with deep learning model

    '''
