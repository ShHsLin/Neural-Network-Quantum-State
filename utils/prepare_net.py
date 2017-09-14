'''
from wavefunction.tf_NN3 import tf_NN3
from wavefunction.tf_CNN import tf_CNN
from wavefunction.tf_FCN import tf_FCN
from wavefunction.tf_NN_complex import tf_NN_complex
from wavefunction.tf_NN3_complex import tf_NN3_complex
from wavefunction.tf_NN_RBM import tf_NN_RBM
'''
from wavefunction.tf_network import tf_network


def prepare_net(which_Net, systemSize, opt, alpha):
    return tf_network(systemSize, optimizer=opt, alpha=alpha)

    if which_Net == "NN":
        Net = tf_NN(systemSize, optimizer=opt, alpha=alpha)
    elif which_Net == "NN3":
        Net = tf_NN3(systemSize, optimizer=opt, alpha=alpha)
    elif which_Net == "CNN":
        Net = tf_CNN(systemSize, optimizer=opt)
    elif which_Net == "FCN":
        Net = tf_FCN(systemSize, optimizer=opt)
    elif which_Net == "NN_complex":
        Net = tf_NN_complex(systemSize, optimizer=opt, alpha=alpha)
    elif which_Net == "NN3_complex":
        Net = tf_NN3_complex(systemSize, optimizer=opt, alpha=alpha)
    elif which_Net == "NN_RBM":
        Net = tf_NN_RBM(systemSize, optimizer=opt)
    elif which_Net == "network":
        Net = tf_network(systemSize, optimizer=opt, alpha=alpha)
    else:
        raise NotImplementedError

    return Net
