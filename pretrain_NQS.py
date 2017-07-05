from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from utils.parse_args import parse_args

from wavefunction.tf_NN import tf_NN
from wavefunction.tf_NN3 import tf_NN3
from wavefunction.tf_CNN import tf_CNN
from wavefunction.tf_FCN import tf_FCN
from wavefunction.tf_NN_complex import tf_NN_complex
from wavefunction.tf_NN3_complex import tf_NN3_complex
from wavefunction.tf_NN_RBM import tf_NN_RBM

if __name__ == "__main__":

    args = parse_args()
    L = args.L
    which_Net = args.which_Net
    lr = args.lr
    batch_size = args.batch_size

    basis = []
    for line in open('EigenVec/basisMatrix'+str(L)+'.csv', 'r'):
        basis.append(line[:-1])

    newbasis = np.zeros((len(basis), L, 2))
    for i in range(len(basis)):
        for j in range(L):
            newbasis[i, j, 0] = basis[i][j]

    systemSize = (L, 2)
    if which_Net == 'NN':
        Net = tf_NN(systemSize, optimizer='Mom', alpha=9)
    elif which_Net == 'NN3':
        Net = tf_NN3(systemSize, optimizer='Mom', alpha=2)
    elif which_Net == 'CNN':
        Net = tf_CNN(systemSize, optimizer='Mom')
    elif which_Net == 'FCN':
        Net = tf_FCN(systemSize, optimizer='Mom')
    elif which_Net == 'NN_complex':
        Net = tf_NN_complex(systemSize, optimizer='Mom', alpha=2)
    elif which_Net == 'NN3_complex':
        Net = tf_NN3_complex(systemSize, optimizer='Mom', alpha=1)
    elif which_Net == 'NN_RBM':
        Net = tf_NN_RBM(systemSize, optimizer='Mom', alpha=2)
    else:
        raise NotImplementedError

    num_train = 2**L
    to_large_dict = {}
    X = np.zeros((num_train, L, 2))
    for i in range(0, num_train):
        temp = i+1
        for j in range(L):
            if temp > 2**(L-j-1):
                X[i, j, 0] = 1
                temp -= 2**(L-j-1)
            else:
                X[i, j, 1] = 1
                pass
        to_large_dict[''.join([str(int(ele)) for ele in X[i, :, 0]])] = i

    X = X.reshape((num_train, L*2))

    # Y = np.zeros((num_train, 1))
    # amp_array = np.genfromtxt(open('EigenVec/EigVec_L'+str(L)+'V20W0E0.csv', 'r'))
    # for i in range(len(amp_array)):
    #    idx = to_large_dict[''.join([str(int(ele)) for ele in newbasis[i, :, 0]])]
    #    Y[idx] = amp_array[i]

    # X_nz = X[(np.abs(Y) * np.ones((1, 32)) >= 1e-15)]
    # Y_nz = Y[(np.abs(Y) >= 1e-15)]
    # X = X_nz.reshape((len(Y_nz), 32))
    # Y = Y_nz.reshape((len(Y_nz),1))
    Y = np.genfromtxt('EigenVec/eig_L'+str(L)+'_PBC.csv').reshape((2**L, 1))
    print X.shape, Y.shape

    with Net.sess as sess:

        true_out = tf.placeholder(tf.float32, [None, 1])
        v1 = true_out
        v2 = Net.pred
        cost = -tf.reduce_sum(tf.multiply(v1, v2))/tf.norm(v1)/tf.norm(v2)
        # cost = -tf.reduce_sum(tf.multiply(true_out, tf.log(Net.pred)))
        # cost = tf.nn.l2_loss((Net.pred - true_out))

        learning_rate = tf.Variable(lr)
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        # train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)

    #    from tensorflow.python import debug as tf_debug
    #    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        sess.run(tf.global_variables_initializer())

        print len(Net.model_var_list), len(Net.para_list)
        saver = tf.train.Saver(Net.model_var_list)  # Net.model_var_list)
        ckpt = tf.train.get_checkpoint_state('Model/'+which_Net+'/L'+str(L))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(Net.sess, ckpt.model_checkpoint_path)
            print("Restore from last check point")
        else:
            print("No checkpoint found")

        min_c = 10000
        #    sess.run(learning_rate.assign(0.1 * (0.8**(i/10000))))
        print("-------- Start training -------\n")
        print("Total num para: ", Net.getNumPara())

        for i in xrange(100000):
            # batch_mask = np.random.choice(len(Y), batch_size)  # ,p=Y*Y)
            p = (np.abs(Y)).flatten()
            p = p + 0.1/batch_size
            p = p/sum(p)
            batch_mask = np.random.choice(len(Y), batch_size, p=p)

            if i % 5000 == 0:
                y = sess.run(Net.pred, feed_dict={Net.x: X})
                print('y norm : ', np.linalg.norm(y))
                print (Y.flatten().dot(y.flatten())/np.linalg.norm(Y)/np.linalg.norm(y))
                # import matplotlib.pyplot as plt
                # fig = plt.figure()
                # plt.plot(Y/np.linalg.norm(Y), '-o')
                # plt.plot(y/np.linalg.norm(y), '--')
                # plt.show()
                pass

            _, c, y = sess.run([train_step, cost, Net.pred], feed_dict={Net.x: X[batch_mask],
                                                                        true_out: Y[batch_mask]})
            # _, c = sess.run([train_step, cost], feed_dict={Net.x: X, true_out: Y})
    #         rad, angle, out, fc1, fc2= sess.run([Net.rad, Net.angle, Net.out, Net.fc1,
    #                                              Net.fc2],feed_dict={Net.x: X[batch_mask]})
    #         print "rad: ", rad
    #         print "angle: ", angle
    #         print "out: ", out
    #         print "fc1: ", fc1
    #         print "fc2: ", fc2

            if i % 500 == 0:
                print("step:", i, " cosine error:", c, "Y norm",
                      np.linalg.norm(Y[batch_mask]), "y norm:", np.linalg.norm(y),
                      "dot:",
                      y.T.dot(Y[batch_mask])[0]/np.linalg.norm(Y[batch_mask])/np.linalg.norm(y))
    #            if min_c > c:
    #                min_c = c
                saver.save(sess, 'Model/'+which_Net+'/L'+str(L)+'/pre')

    # fig.savefig('L16_pretrain.eps',bbox_inches='tight')
