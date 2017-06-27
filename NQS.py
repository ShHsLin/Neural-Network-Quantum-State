from __future__ import print_function
import numpy as np
import pickle
import tensorflow as tf
from tf_FCN import tf_FCN
from tf_CNN import tf_CNN
from tf_NN import tf_NN
from tf_NN3 import tf_NN3


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def read_object(filename):
    with open(filename, 'r') as input:
        return pickle.load(input)

# TASK!! #


"""
1.  Should move config out as an indep class
So that easily to change from 1d problem to 2d problem?
2.  Rewrite the h, J,... etc in a class model
So easily to switch model
"""


class NQS():
    def __init__(self, inputShape, Net, Hamiltonian):
        self.config = np.zeros((inputShape[0], inputShape[1], 1), dtype=int)
        self.config[:, 1, 0] = 1
        self.NNet = Net
        self.moving_E_avg = None

        self.ampDic = {}
        if Hamiltonian == 'Ising':
            self.get_local_E = self.local_E_Ising
        elif Hamiltonian == 'AFH':
            self.get_local_E = self.local_E_AFH

    def getNumPara(self):
        return self.NNet.getNumPara()

    def getSelfAmp(self):
        configStr = ''.join([str(ele) for ele in self.config.flatten()])
        if configStr in self.ampDic:
            return self.ampDic[configStr]
        else:
            amp = float(self.NNet.forwardPass(self.config))
            self.ampDic[configStr] = amp
            return amp

    def eval_amp_array(self, configArray):
        inputShape0, inputShape1, numData = configArray.shape
        ampArray = np.zeros((numData))
        for i in range(numData):
            config = configArray[:, :, i]
            configStr = ''.join([str(ele) for ele in config.flatten()])
            if configStr in self.ampDic:
                amp = self.ampDic[configStr]
            else:
                amp = float(self.NNet.forwardPass(self.config))
                self.ampDic[configStr] = amp

            ampArray[i] = amp

        return ampArray

    def cleanAmpDic(self):
        self.ampDic = {}

    def newconfig(self):
        L = self.config.shape[0]
        randsite = np.random.randint(L)
        tempconfig = self.config.copy()
        tempconfig[randsite, :, 0] = (tempconfig[randsite, :, 0] + 1) % 2
        ratio = self.NNet.forwardPass(tempconfig)[0] / self.getSelfAmp()
        if np.random.rand() < np.amin([1., ratio**2]):
            self.config = tempconfig
        else:
            pass

    def H_exp(self, numSample=1000, h=1, J=0):
        energyList = []
        correlength = 10
        for i in range(numSample * correlength):
            self.newconfig()
            if i % correlength == 0:
                _, _, localEnergy, _ = self.getLocal(h, J)
                energyList = np.append(energyList, localEnergy)
            else:
                pass
        print(energyList)
        return np.average(energyList)

    def VMC(self, numSample, iteridx=0):
        self.cleanAmpDic()

        L = self.config.shape[0]
        numPara = self.getNumPara()
        OOsum = np.zeros((numPara, numPara))
        Osum = np.zeros((numPara))
        Earray = []
        EOsum = np.zeros((numPara))
        # localO,localOO,localE,localEO = self.getLocal(h,J)
        # OOsum	=	np.zeros(localOO.shape)
        # Osum	=	np.zeros(localO.shape)
        # Earray	=	[]
        # EOsum	=	np.zeros(localEO.shape)

        corrlength = 10
        configDim = list(self.config.shape)
        configDim[2] = numSample
        configArray = np.zeros(configDim)
        for i in range(numSample * corrlength):
            self.newconfig()
            if i % corrlength == 0:
                configArray[:, :, i / corrlength] = self.config[:, :, 0]
                localO, localOO, localE, localEO = self.getLocal()
                OOsum += localOO
                Osum += localO
                Earray.append(localE)
                EOsum += localEO
            else:
                pass

        # ## If batch ###########################
        #        localO,localOO,localE,localEO = self.getLocalBatch(configArray,h=h,J=J)
        #        OOsum   =   np.einsum('ijk->ij',localOO)
        #        Osum    =   np.einsum('ij->i' ,localO)
        #        Earray  =   localE
        #        EOsum   =   np.einsum('ij->i' ,localEO)
        # ##### Have not finish batch below #####

        Earray = np.array(Earray).flatten()
        Eavg = np.average(Earray)
        Evar = np.var(Earray)
        print(self.getSelfAmp())
        print("E/N !!!!: ", Eavg / L, "  Var: ", Evar / L)  # , "Earray[:10]",Earray[:10]
        # S_ij = <O_i O_j > - <O_i><O_j>
        Sij = OOsum / numSample - np.einsum('i,j->ij', Osum.flatten(), Osum.flatten()) / (numSample**2)
        ############
        # Method 1 #
        ############
        # regu_para = np.amax([10 * (0.9**iteridx), 1e-4])
        # Sij = Sij + regu_para * np.diag(np.ones(Sij.shape[0]))
        # invSij = np.linalg.inv(Sij)
        ############
        # Method 2 #
        ############
        Sij = Sij+np.diag(np.ones(Sij.shape[0])*1e-10)
        invSij = np.linalg.pinv(Sij, 1e-2)
        # Fj = 2<O_iH>-2<H><O_i>
        if self.moving_E_avg is None:
            Fj = 2. * (EOsum / numSample - Eavg * Osum / numSample)
        else:
            self.moving_E_avg = self.moving_E_avg * 0.5 + Eavg * 0.5
            Fj = 2. * (EOsum / numSample - self.moving_E_avg * Osum / numSample)
            print("moving_E_avg/N !!!!: ", self.moving_E_avg / L)

        Gj = invSij.dot(Fj.T)
        print(np.linalg.norm(Gj), "norm(F):", np.linalg.norm(Fj))

        return Gj, configArray, Eavg / L

    def getLocal(self):
        Wsize = self.getNumPara()
        localO = np.zeros((Wsize))
        localOO = np.zeros((Wsize, Wsize))
        localE = 0.
        localEO = np.zeros((Wsize))

        config = self.config
        localE = self.get_local_E(config)

        GList = self.NNet.backProp(self.config)
        localOind = 0
        for i in range(len(GList)):
            G = GList[i].flatten()
            localO[localOind: localOind + G.size] = G
            localOind += G.size

        localOO = np.einsum('i,j->ij', localO.flatten(), localO.flatten())
        localEO = localO * localE

        return localO, localOO, localE, localEO

    # local_E_Ising, get local E from Ising Hamiltonian
    # For only one config.
    def local_E_Ising(self, config, h=1):
        L, inputShape1, numData = config.shape
        localE = 0.
        for i in range(L - 1):
            temp = config[i, :, 0].dot(config[i + 1, :, 0])
            localE -= 2 * (temp - 0.5)

        # Periodic Boundary condition
        temp = config[0, :, 0].dot(config[-1, :, 0])
        localE -= 2 * (temp - 0.5)
        #####################################

        oldAmp = self.eval_amp_array(config)[0]
        for i in range(L):
            tempConfig = config.copy()
            tempConfig[i, :] = (tempConfig[i, :] + 1) % 2
            tempAmp = float(self.NNet.forwardPass(tempConfig))
            localE -= h * tempAmp / oldAmp

        return localE

    def local_E_AFH(self, config, J=1):
        L, inputShape1, numData = config.shape
        localE = 0.
        oldAmp = self.eval_amp_array(config)[0]
        for i in range(L - 1):
            temp = config[i, :, 0].dot(config[i + 1, :, 0])
            localE += 2 * (temp - 0.5) * J / 4
            if config[i, :, 0].dot(config[i + 1, :, 0]) == 0:
                tempConfig = config.copy()
                tempConfig[i, :] = (tempConfig[i, :] + 1) % 2
                tempConfig[i + 1, :] = (tempConfig[i + 1, :] + 1) % 2
                tempAmp = float(self.NNet.forwardPass(tempConfig))
                localE += J * tempAmp / oldAmp / 2
            else:
                pass
        '''
        Periodic Boundary condition
        '''
        temp = config[0, :, 0].dot(config[-1, :, 0])
        localE += 2 * (temp - 0.5) * J / 4
        if temp == 0:
            tempConfig = config.copy()
            tempConfig[i, :] = (tempConfig[i, :] + 1) % 2
            tempConfig[i + 1, :] = (tempConfig[i + 1, :] + 1) % 2
            tempAmp = float(self.NNet.forwardPass(tempConfig))
            localE += J * tempAmp / oldAmp / 2

        return localE

########################
#  END OF DEFINITION  #
########################


L = 10
systemSize = (L, 2)
which_Net = 'NN3'
opt = 'Mom'
H = 'AFH'

if which_Net == "NN":
    Net = tf_NN(systemSize, optimizer=opt, alpha=10)
elif which_Net == "NN3":
    Net = tf_NN3(systemSize, optimizer=opt, alpha=2)
elif which_Net == "CNN":
    Net = tf_CNN(systemSize, optimizer=opt)
elif which_Net == "FCN":
    Net = tf_FCN(systemSize, optimizer=opt)

N = NQS(systemSize, Net=Net, Hamiltonian=H)

var_shape_list = [var.get_shape().as_list() for var in N.NNet.para_list]
var_list = tf.global_variables()
saver = tf.train.Saver(N.NNet.model_var_list)
ckpt = tf.train.get_checkpoint_state('Model/'+which_Net+'/L'+str(L)+'/')

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(N.NNet.sess, ckpt.model_checkpoint_path)
    print("Restore from last check point")
else:
    print("No checkpoint found")


# Thermalization
for i in range(1000):
    N.newconfig()

E_log = []
# wc1 = np.load('wc1.npy')
# bc1 = np.load('bc1.npy')
N.NNet.sess.run(N.NNet.learning_rate.assign(1e-1))
N.NNet.sess.run(N.NNet.momentum.assign(0.5))
_, _, E_avg = N.VMC(numSample=1500, iteridx=0)
# N.moving_E_avg = E_avg * l

for iteridx in range(0, 1500):
    print(iteridx)
    # N.NNet.sess.run(N.NNet.weights['wc1'].assign(wc1))
    # N.NNet.sess.run(N.NNet.biases['bc1'].assign(bc1))

    N.NNet.sess.run(N.NNet.learning_rate.assign(1e-2 * (0.995**iteridx)))
    N.NNet.sess.run(N.NNet.momentum.assign(0.95 - 0.4 * (0.98**iteridx)))
    numSample = 3000  # + iteridx
    GradW, batchConfig, E = N.VMC(numSample=numSample, iteridx=iteridx)
    # GradW = GradW/np.linalg.norm(GradW)*np.amax([(0.95**iteridx),0.1])
    E_log.append(E)
    grad_list = []
    grad_ind = 0
    for var_shape in var_shape_list:
        var_size = np.prod(var_shape)
        grad_list.append(GradW[grad_ind:grad_ind + var_size].reshape(var_shape))
        grad_ind += var_size

    N.NNet.applyGrad(grad_list)
#  L2 Regularization ###
#    for idx, W in enumerate( N.NNet.sess.run(N.NNet.para_list)):
    #        GList[idx] += W*0.1


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
