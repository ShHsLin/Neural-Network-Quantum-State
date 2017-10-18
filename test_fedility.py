from __future__ import absolute_import
import os
import tensorflow as tf
import numpy as np

from utils.parse_args import parse_args
from network.tf_network import tf_network
import network.tf_wrapper as tf_


if __name__ == "__main__":

    J2 = 0
    L = 16
    opt = 'Mom'
    batch_size = 100
    system_size = (L, 2)
    Net = tf_network('sRBM', system_size, optimizer='Mom', dim=1, alpha=4)
    print("Total num para: ", Net.getNumPara())

    basis = []
    for line in open('EigenVec/Sz0_basisMatrix'+str(L)+'.csv', 'r'):
        basis.append(line[:-1])

    newbasis = np.zeros((len(basis), L, 2))
    for i in range(len(basis)):
        for j in range(L):
            newbasis[i, j, 0] = basis[i][j]

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

    Y = np.genfromtxt('EigenVec/ES_L16_J2_%d.csv' % J2).reshape((2**L, 1))
    print X.shape, Y.shape
    mask = (np.einsum('ij->i',X[:,:,0])==8)
    X = X[mask, :,:]
    Y = Y[mask]
    print X.shape, Y.shape

    fidelity_list = []
    with Net.sess as sess:
        for idx in [52,
                    86,
                    34,
                    43,
                    14,
                    49,
                    99,
                    58,
                    78,
                    37
                    ]:# range(1, 101):
            true_out = tf.placeholder(tf.float32, [None, 1])
            v1 = true_out
            v2 = Net.pred
            cost = -tf.reduce_sum(tf.multiply(v1, v2))/tf.norm(v1)/tf.norm(v2)
            # cost = -tf.reduce_sum(tf.multiply(true_out, tf.log(Net.pred)))
            # cost = tf.nn.l2_loss((Net.pred - true_out))
            lr = 1e-3
            learning_rate = tf.Variable(lr)
            Optimizer = tf_.select_optimizer(optimizer=opt, learning_rate=learning_rate,
                                             momentum=0.9)
            train_step = Optimizer.minimize(cost)

            # from tensorflow.python import debug as tf_debug
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            sess.run(tf.global_variables_initializer())

            print len(Net.model_var_list), len(Net.para_list)
            saver = tf.train.Saver(Net.model_var_list)
            ckpt_path = '../Job_Result/J1J2.%d/L16_sRBM_a4_Mom1e-03_S1000/%d/wavefunction/vmc1d/sRBM/L16a4/' % (J2, idx)
            if not os.path.exists(ckpt_path):
                raise
                os.makedirs(ckpt_path)
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(Net.sess, ckpt.model_checkpoint_path)
                print("Restore from last check point")
            else:
                print("No checkpoint found")

            p = (np.abs(Y)).flatten()
            p = p + 0.1/batch_size
            p = p/sum(p)

            total_cos_accu = []
            batch_cos_accu = []
            # import pdb;pdb.set_trace()

            for i in xrange(0+1):
                # batch_mask = np.random.choice(len(Y), batch_size)  # ,p=Y*Y)
                batch_mask = np.random.choice(len(Y), batch_size, p=p)

                if i % 5000 == 0:
                    y = sess.run(Net.pred, feed_dict={Net.x: X})
                    print('y norm : ', np.linalg.norm(y))
                    c = Y.flatten().dot(y.flatten())/np.linalg.norm(Y)/np.linalg.norm(y)
                    print(c)
                    fidelity_list.append(c)
                    PLOT = False
                    if PLOT:
                        import matplotlib.pyplot as plt
                        fig = plt.figure()
                        plt.plot(Y/np.linalg.norm(Y), '-o')
                        plt.plot(y/np.linalg.norm(y), '--')
                        plt.show()
                    pass

                _, c, y = sess.run([train_step, cost, Net.pred], feed_dict={Net.x: X[batch_mask],
                                                                            true_out: Y[batch_mask]})
                batch_cos_accu.append(-c)

        # np.savetxt('log/pretrain/L%d_%s_a%s_%s%.e_batch.csv' % (L, which_net, alpha, opt, lr),
        #            batch_cos_accu, '%.4e', delimiter=',')
        # np.savetxt('log/pretrain/L%d_%s_a%s_%s%.e_total.csv' % (L, which_net, alpha, opt, lr),
        #            total_cos_accu, '%.4e', delimiter=',')

        # fig.savefig('L16_pretrain.eps',bbox_inches='tight')

print(np.sort(fidelity_list))
