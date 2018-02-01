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
import NQS


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
        N = NQS.NQS_1d(systemSize, Net=Net, Hamiltonian=H, batch_size=batch_size)
    else:
        N = NQS.NQS_2d(systemSize, Net=Net, Hamiltonian=H, batch_size=batch_size)

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
    GradW = None
    # N.moving_E_avg = E_avg * l
    SzSz = N.VMC_observable(num_sample=num_sample)

    log_file = open('SzSz_L%d_%s_a%s_%s%.e_S%d.csv' % (L, which_net, alpha, opt, lr, num_sample),
                    'a')
    np.savetxt(log_file, SzSz, '%.4e', delimiter=',')
    log_file.close()
